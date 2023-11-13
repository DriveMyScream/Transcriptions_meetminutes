import boto3
import requests
import json
import time
import uuid
import shutil
from botocore.exceptions import ClientError
from pydub import AudioSegment
from pathlib import Path
import concurrent.futures
import pandas as pd
import assemblyai as aai
from pydub.silence import detect_silence


RECALL_AUTH_TOKEN = 'Token fc85b6a828118eb82de854d2d67140837a7a47fa'
aai.settings.api_key = '04e7afe730d24159bdb79bbb33a59f20'

# Initialize Resources
transcriber = aai.Transcriber()  # More configuration in the class
# boto3.resource are easy to code than client
s3_resource = boto3.resource('s3')
s3_client = boto3.client('s3')
# boto3.resource doesn't provide api for all, so client has to be used
transcribe_client = boto3.client('transcribe', region_name='ap-south-1')

# Language configs


def load_language_configuration():
    with open("language_config.json", "r") as file:
        return json.load(file)


language_configuration = load_language_configuration()


def clean_timeline(timeline):
    """
    Handles the negative timestamps and the timeline starts from 0
    """
    cleaned_timeline = []
    negative_timeline = []
    for i in timeline:
        if i['timestamp'] < 0:
            negative_timeline.append(i)  # store negative timestamps separately
        else:
            cleaned_timeline.append(i)  # store positive timestamps
    # sort negative timestamps in reverse order to get the latest one first
    negative_timeline.sort(key=lambda x: x['timestamp'], reverse=True)
    if negative_timeline:
        # take the latest negative timestamp and change it to 0
        latest_negative = negative_timeline[0]
        latest_negative['timestamp'] = 0
        # insert this at the beginning of cleaned_timeline
        cleaned_timeline.insert(0, latest_negative)
    return cleaned_timeline


def get_timeline(bot_id):
    url = f"https://api.recall.ai/api/v1/bot/{bot_id}/speaker_timeline/"
    payload = {}
    headers = {'Authorization': RECALL_AUTH_TOKEN}
    response = requests.request("GET", url, headers=headers, data=payload)
    timeline = response.json()
    timeline = clean_timeline(timeline)
    return timeline


def get_video_url(bot_id):  # Expecting a valid bot_id
    url = f"https://api.recall.ai/api/v1/bot/{bot_id}/"
    payload = {}
    headers = {'Authorization': RECALL_AUTH_TOKEN}
    response = requests.request("GET", url, headers=headers, data=payload)
    video_url = response.json()['video_url']
    return video_url


def start_aws_job_multi_lingual(job_name, media_uri, language_config):
    vocabulary_name = None
    try:
        job_args = {
            'TranscriptionJobName': job_name,
            'Media': {'MediaFileUri': media_uri},
            'MediaFormat': 'wav',
            'IdentifyMultipleLanguages': True,
            'LanguageOptions': language_config,
            'Settings': {'ShowSpeakerLabels': True,
                         'MaxSpeakerLabels': 10}}  # https://docs.aws.amazon.com/transcribe/latest/APIReference/API_StartTranscriptionJob.html#transcribe-StartTranscriptionJob-request-IdentifyMultipleLanguages
        if vocabulary_name is not None:
            job_args['Settings'] = {'VocabularyName': vocabulary_name}
        response = transcribe_client.start_transcription_job(**job_args)
        job = response['TranscriptionJob']
    except ClientError:
        raise
    else:
        return job


def start_aws_job(job_name, media_uri, language_config):
    vocabulary_name = 'simple_vocabulary'
    try:
        job_args = {
            'TranscriptionJobName': job_name,
            'Media': {'MediaFileUri': media_uri},
            'MediaFormat': 'wav',
            'LanguageCode': language_config,
            'IdentifyLanguage': False
        }
        if vocabulary_name is not None:
            response = transcribe_client.create_vocabulary(
                LanguageCode = language_config,
                VocabularyName = vocabulary_name,
                Phrases = ['meetminutes']
            )
            while True:
                status = transcribe_client.get_vocabulary(VocabularyName = vocabulary_name)
                if status['VocabularyState'] in ['READY', 'FAILED']:
                    break
                print("Not ready yet...")
                time.sleep(5)

#        job_args['Settings'] = {'VocabularyName': vocabulary_name}
        response = transcribe_client.start_transcription_job(**job_args)
        job = response['TranscriptionJob']
    except ClientError:
        raise
    else:
        return job


def get_aws_job(job_name):
    try:
        response = transcribe_client.get_transcription_job(
            TranscriptionJobName=job_name)
        job = response['TranscriptionJob']
    except ClientError:
        raise
    else:
        return job


def get_speaker_from_timestamp(timestamp, timeline):
    """
    Retrieve the speaker timeline element corresponding to the given timestamp using binary search.
    Args:
    - timestamp (float): The timestamp for which to retrieve the speaker.
    - timeline (list): The timeline information from recall.
    Returns:
    - dict: The timeline element corresponding to the timestamp.
    """
    low, high = 0, len(timeline) - 1
    while low <= high:
        mid = (low + high) // 2
        # If the timestamp is less than the current mid timestamp
        if timestamp < timeline[mid]['timestamp']:
            high = mid - 1
        # If the timestamp is greater than or equal to the current mid timestamp
        elif mid + 1 < len(timeline) and timestamp >= timeline[mid+1]['timestamp']:
            low = mid + 1
        else:
            return timeline[mid]
    # Return None if no match is found
    return None


def transform_punctuation(df):
    df_copy = df.copy()
    for index, row in df_copy.iterrows():
        if row['type'] == 'punctuation':
            # If the current row is the first row, skip it (because there's no previous row to modify)
            if index == 0:
                continue
            # Append punctuation to the content of the previous row
            df_copy.at[index-1, 'content'] = df_copy.at[index -
                                                        1, 'content'] + row['content']
            # Set the content of the current row to NaN (we'll drop these rows later)
            df_copy.at[index, 'content'] = float('nan')
    # Drop rows with NaN in the 'content' column (these are the punctuation rows we modified)
    df_cleaned = df_copy.dropna(subset=['content']).reset_index(drop=True)
    return df_cleaned[['start_time', 'end_time', 'content', 'speaker']]


def convert_to_vtt_time(seconds):
    """Convert seconds to WebVTT timestamp format."""
    hour, minute, second = int(
        seconds // 3600), int((seconds % 3600) // 60), seconds % 60
    return "{:02}:{:02}:{:06.3f}".format(hour, minute, second)


class MeetMinutesTranscribe:
    def __init__(self, bot_id) -> None:
        self.bot_id = bot_id
        self.job_name = bot_id
        self.media_dir = '/tmp'/Path(bot_id)
        self.media_dir.mkdir(parents=True, exist_ok=True)
        self.video_path = self.media_dir / f"{bot_id}.mp4"
        self.audio_path = self.media_dir / f"{bot_id}.wav"
        self.no_silence_audio_path = self.media_dir / \
            f"{bot_id}_no_silence.wav"
        self.sample_rate = 44100  # Hz
        self.reduced_sample_rate = 100  # Hz for faster silence detection
        self.silence_threshold = -50
        self.min_silence_len = 5000
        self.chunk_duration = 120  # sec
        self.aws_transcribe_polling_frequency = 3  # sec
        self.num_parallel_transcription = 8
        self.input_file_format = "mp4"
        self.output_file_format = "wav"
        self.bucket_name = 'meetminutes'
        self.timeline = get_timeline(bot_id)
        self.chunked_timeline = self.chunk_timeline()
        self.is_long_audio = False
        self.vocabulary_word_boost = []
        self.vocabulary_boost_param = 'default'

    def download_video(self):
        video_url = get_video_url(self.bot_id)
        r = requests.get(video_url)
        with open(self.video_path, 'wb') as f:
            f.write(r.content)

    def mp4_to_wav_convert(self):
        """
        Converts mp4 to mono WAV audio using pydub.
        """
        audio = AudioSegment.from_file(
            self.video_path, format=self.input_file_format)
        resampled_mono_audio = audio.set_channels(
            1).set_frame_rate(self.sample_rate)
        resampled_mono_audio.export(self.audio_path.with_suffix(
            '.'+self.output_file_format), format=self.output_file_format)
        return self.audio_path.with_suffix('.'+self.output_file_format)

    def audio_duration(self):
        audio = AudioSegment.from_file(self.video_path, format="mp4")
        audio_duration_in_milliseconds = len(audio)
        self.audio_duration_in_seconds = audio_duration_in_milliseconds / 1000.0
        if self.audio_duration_in_seconds > 10800:  # 3 hours in seconds
            self.is_long_audio = True
        return self.audio_duration_in_seconds

    def chunk_timeline(self):
        '''
        Each chunk has guaranteed "chunk_duration" seconds or more seonds of audio, except the last
        '''
        chunks_list = []  # A list of chunks
        start_timestamp = self.timeline[0]['timestamp']
        chunk = []  # Each chunk of timestamps
        for item in self.timeline:
            timestamp = item['timestamp']
            chunk.append(item)
            # Check if the difference between the current timestamp and the start timestamp exceeds the chunk_duration
            if timestamp - start_timestamp >= self.chunk_duration:
                # If the duration exceeds, append chunk to chunk_list
                chunks_list.append(chunk)
                start_timestamp = timestamp
                chunk = []
        # Final append for remaing chunk
        if chunk:
            chunks_list.append(chunk)
        return chunks_list

    def split_audio(self):
        # After the split, offsets and chunked_paths will be available
        audio_split_points = [float(i[0]['timestamp'])
                              for i in self.chunked_timeline]
        audio = AudioSegment.from_file(self.audio_path)
        chunks = []
        chunked_paths = []
        start_point = int(audio_split_points[0] * 1000)
        for split_point in audio_split_points[1:]:
            # For first audio, end_point is split_point[i+1]
            end_point = int(split_point * 1000)
            chunks.append(audio[start_point:end_point])  # Ordered
            start_point = end_point  # For next iteration, this is the new head
        chunks.append(audio[start_point:])  # final element after iteration
        for i, chunk in enumerate(chunks):
            output_path = self.media_dir / Path(f"audio_split_{i}.wav")
            chunk.export(str(output_path), format="wav")
            chunked_paths.append(str(output_path))
        self.offsets = audio_split_points
        self.chunked_paths = chunked_paths
        # Initialize transcriptions to maintain order later
        # self.standard_transcription_result = [None]*len(self.offsets)
        # self.final_transcription_result = [None]*len(self.offsets)

    def s3_upload_chunks(self, file, object_name):
        '''For AWS Transcribe ONLY'''
        s3_resource.Object(self.bucket_name,
                           f'temp/{self.bot_id}/{object_name}').put(Body=file)
        s3_uri = f's3://meetminutes/temp/{self.bot_id}/{object_name}'
        return s3_uri

    def aws_multilingual_transcription(self, i):
        with open(self.chunked_paths[i], 'rb') as file:
            filename = f"{self.bot_id}_segment_{self.offsets[i]}.wav"
            uri = self.s3_upload_chunks(file, filename)
        job_name = str(uuid.uuid4())
        job_details = start_aws_job_multi_lingual(
            # Set language via set_language_options
            job_name, uri, language_config=self.language_config)
        # Could possibly use job_details for logging
        job = get_aws_job(job_name)
        while job['TranscriptionJobStatus'] == 'IN_PROGRESS':
            time.sleep(self.aws_transcribe_polling_frequency)
            job = get_aws_job(job_name)
        transcript_response = requests.get(
            job['Transcript']['TranscriptFileUri'])
        transcription = transcript_response.json()["results"]['items']
        return transcription

    def aws_transcription(self, i):
        with open(self.chunked_paths[i], 'rb') as file:
            filename = f"{self.bot_id}_segment_{self.offsets[i]}.wav"
            uri = self.s3_upload_chunks(file, filename)
        job_name = str(uuid.uuid4())
        job_details = start_aws_job(job_name, uri, self.language_config)
        job = get_aws_job(job_name)
        while job['TranscriptionJobStatus'] == 'IN_PROGRESS':
            time.sleep(self.aws_transcribe_polling_frequency)
            job = get_aws_job(job_name)
        transcript_response = requests.get(
            job['Transcript']['TranscriptFileUri'])
        transcription = transcript_response.json()["results"]['items']
        return transcription

    def assembly_transcription(self, i):
        config = aai.TranscriptionConfig(language_code=self.language_config,
                                         punctuate=False,
                                         format_text=False,
                                         word_boost=self.vocabulary_word_boost,
                                         boost_param=self.vocabulary_boost_param)
        print('Boosted vocab: ', self.vocabulary_word_boost, self.vocabulary_boost_param)
        transcriber = aai.Transcriber(config=config)
        transcript = transcriber.transcribe(self.chunked_paths[i])
        transcription = [dict(i) for i in transcript.words]
        return transcription

    def convert_assembly_to_standard(self):
        '''Applied offsets also (remider since it could be common bug when implemending a new provider)
        The error could come in get_timeline and have to debug it back to a converter. So very important!
        '''
        transcription_list = []
        for i in range(len(self.transcription_responses)):
            transcription = pd.DataFrame(self.transcription_responses[i])
            print(transcription.shape)
            if transcription.shape!=(0,0):
                transcription = transcription.rename(columns={
                                                    'text': 'content', 'start': 'start_time', 'end': 'end_time'}).drop('confidence', axis=1)

                def ms_to_s(milliseconds): return int(milliseconds)/1000
                transcription['start_time'] = transcription.start_time.apply(
                    ms_to_s)
                transcription['start_time'] = transcription.start_time.apply(
                    lambda x: x+float(self.offsets[i]))
                transcription['end_time'] = transcription.end_time.apply(ms_to_s)
                transcription['end_time'] = transcription.end_time.apply(
                    lambda x: x+float(self.offsets[i]))
                transcription['speaker'] = transcription.start_time.apply(
                    lambda x: get_speaker_from_timestamp(x, self.chunked_timeline[i])['name'])
                transcription_list.append(transcription)
        self.transcription_list = transcription_list
        self.standard_df = pd.concat(transcription_list).reset_index(drop=True)
        self.output_wljson = json.dumps(self.standard_df.to_dict(orient='list'), ensure_ascii=False)

    def set_transcription_mode(self, internal_language_code):
        self.internal_language_code = internal_language_code
        self.transcribe_mode, self.language_config = language_configuration[
            self.internal_language_code]
        # Supports multiple language, average quality, more languages
        if self.transcribe_mode.lower().strip() == 'aws-transcribe-multi':
            self.transcription_function = self.aws_multilingual_transcription
            self.adapt_transcription = self.convert_aws_to_standard
        # Supports single language, average quality, more languages
        elif self.transcribe_mode.lower().strip() == 'aws-transcribe':
            self.transcription_function = self.aws_transcription
            self.adapt_transcription = self.convert_aws_to_standard
        # Support single language, good quality, less languages
        elif self.transcribe_mode.lower().strip() == 'assemblyai':
            self.transcription_function = self.assembly_transcription
            self.adapt_transcription = self.convert_assembly_to_standard
        else:
            print('Invalid transcribe mode')
            return False

    def set_vocabulary(self, vocabulary_settings):
        try:
            if self.internal_language_code in ['hindi', 'english']:
                self.vocabulary_word_boost = vocabulary_settings['word_boost']
                self.vocabulary_boost_param = vocabulary_settings['boost_param']
                print('Word Boost Set')
        except Exception as e:
            print(str(e))
            self.vocabulary_word_boost = ['MeetMinutes']
            self.vocabulary_boost_param = 'default'
            print('Word Boost Failed')

    def concurrent_transcription(self):
        raw_result = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(self.transcription_function, i): i for i in range(
                len(self.chunked_paths))}
            for future in concurrent.futures.as_completed(futures):
                index = futures[future]
                try:
                    raw_result[index] = future.result()
                except Exception as e:
                    print(f"Thread {index} raised an exception: {e}")
        sorted_results = dict(sorted(raw_result.items()))
        self.transcription_responses = list(sorted_results.values())
        print('concurrent transcription done')

    def convert_aws_to_standard(self):
        transcription_list = []
        for i in range(len(self.transcription_responses)):
            transcription = pd.DataFrame(self.transcription_responses[i])
            if transcription.shape != (0,0):
                transcription['start_time'] = transcription.start_time.astype(
                    'float').apply(lambda x: x+float(self.offsets[i]))
                transcription['end_time'] = transcription.end_time.astype(
                    'float').apply(lambda x: x+float(self.offsets[i]))
                transcription['content'] = transcription.alternatives.apply(
                    lambda x: x[0]['content'])
                transcription['speaker'] = transcription.start_time.apply(
                    lambda x: get_speaker_from_timestamp(x, self.chunked_timeline[i])['name'])
                transcription = transform_punctuation(transcription)
                transcription_list.append(transcription)
        self.standard_df = pd.concat(transcription_list).reset_index(drop=True)
        # Save the standard_df for debugging
        # self.standard_df.to_csv(f"{self.bot_id}.csv", index=False)
        self.output_wljson = json.dumps(self.standard_df.to_dict(orient='list'), ensure_ascii=False)
        print('converted aws to standard')

    def generate_vtt(self):
        """Generate WebVTT formatted content from the DataFrame with updated format and combined words."""
        vtt_content = "WEBVTT\n\n"
        current_speaker = None
        combined_content = ""
        start_time = None
        end_time = None
        for index, row in self.standard_df.iterrows():
            if current_speaker == None:
                current_speaker = row['speaker']
                start_time = convert_to_vtt_time(row['start_time'])
            # If the current speaker is not the same as the previous speaker or end of DataFrame
            if current_speaker != row['speaker']:
                end_time = convert_to_vtt_time(prev_row['end_time'])
                vtt_content += f"{start_time} --> {end_time}\n"
                vtt_content += f"<v {current_speaker}>{combined_content.strip()}</v>\n\n"
                combined_content = ""
                start_time = convert_to_vtt_time(row['start_time'])
                current_speaker = row['speaker']
            combined_content += row['content'] + " "
            prev_row = row
        # Handle the content of the last row after loop ends
        end_time = convert_to_vtt_time(prev_row['end_time'])
        vtt_content += f"{start_time} --> {end_time}\n"
        vtt_content += f"<v {current_speaker}>{combined_content.strip()}</v>\n\n"
        self.output_vtt = vtt_content

    def upload_transcripts_to_s3(self):
        """Uploads a string content as .vtt/.json file to S3 private bucket."""
        content, format = self.output_vtt, 'vtt'
        object_name = f'{self.bot_id}/{self.bot_id}_{self.internal_language_code}.{format}'
        s3_resource.Object(
            self.bucket_name, f'bot_files/{object_name}').put(Body=content)
        content, format = self.output_wljson, 'json'
        object_name = f'{self.bot_id}/{self.bot_id}_{self.internal_language_code}.{format}'
        s3_resource.Object(
            self.bucket_name, f'bot_files/{object_name}').put(Body=content)

    def upload_metadata_to_s3(self):
        """Uploads a string content as .vtt/.json file to S3 private bucket."""
        content = json.dumps(
            {"audio_duration_in_seconds": self.audio_duration_in_seconds, "is_long_audio": self.is_long_audio})
        object_name = f'{self.bot_id}/metadata_{self.bot_id}.json'
        s3_resource.Object(
            self.bucket_name, f'bot_files/{object_name}').put(Body=content)

    def check_file_exists(self, filename):
        key = f'bot_files/{self.bot_id}/{filename}'
        try:
            s3_client.head_object(Bucket='meetminutes', Key=key)
            return True
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                return False
            else:
                print(f"Unexpected error: {e}")
                return False
        except Exception as e:
            print(f"An error occurred: {e}")
            return False

    def is_duplicate_job(self):
        file_checks = [self.check_file_exists(f"metadata_{self.bot_id}.json"),
                       self.check_file_exists(
                           f"{self.bot_id}_{self.internal_language_code}.json"),
                           self.check_file_exists(f"{self.bot_id}_{self.internal_language_code}.vtt")]
        is_duplicate = all(file_checks)
        print(is_duplicate, file_checks)
        return is_duplicate

    def download_file_as_string(self, filename):
        '''Helper fucntion for fetch data from s3'''
        try:
            key = f'bot_files/{self.bot_id}/{filename}'
            response = s3_client.get_object(Bucket="meetminutes", Key=key)
            file_content = response['Body'].read().decode('utf-8')
            return file_content
        except ClientError as e:
            print(f"Unexpected error: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")
        return None

    def fetch_data_from_s3(self):
        metadata = json.loads(self.download_file_as_string(
            f"metadata_{self.bot_id}.json"))
        self.audio_duration_in_seconds, self.is_long_audio = metadata[
            'audio_duration_in_seconds'], metadata['is_long_audio']
        self.output_vtt = self.download_file_as_string(
            f"{self.bot_id}_{self.internal_language_code}.vtt")
        self.output_wljson = self.download_file_as_string(
            f"{self.bot_id}_{self.internal_language_code}.json")

    def clear_media_directory(self):
        if self.media_dir.exists() and self.media_dir.is_dir():
            # Remove the directory and its contents
            shutil.rmtree(self.media_dir)

    def clear_s3_temp_directory(self):
        # Get the bucket resource
        bucket = s3_resource.Bucket(self.bucket_name)
        # List all objects in the folder using the resource interface
        objects = [obj for obj in bucket.objects.filter(
            Prefix=f'temp/{self.bot_id}/')]
        # Check if the folder is empty
        if not objects:
            return
        # Delete all objects in the folder
        for obj in objects:
            obj.delete()
