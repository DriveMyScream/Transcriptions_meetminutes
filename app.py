from utils import MeetMinutesTranscribe
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal
from mangum import Mangum

app = FastAPI()

# Pydantic templates


class InputData(BaseModel):
    bot_id: str
    internal_language_code: Literal['english', 'english-us','english-uk', 'english-au',
                                    'hindi', 'spanish', 'french',
                                    'german', 'italian', 'portuguese', 'dutch',
                                    'japanese', 'chinese', 'finnish', 'korean',
                                    'polish', 'russian', 'turkish', 'ukrainian',
                                    'vietnamese',
                                    'arabic', 'multi_ar_en_hi', 'multi_en_hi']
    vocabulary_settings: dict
    transcription_format: Literal["vtt", "json"]


def transcribe(bot_id, internal_language_code, transcription_format, vocabulary_settings):
    # Initializing the logger and the transcriber
    print(f"Initializing MeetMinutesTranscribe with bot_id : {bot_id}")
    mmu = MeetMinutesTranscribe(bot_id)

    mmu.set_transcription_mode(internal_language_code)
    print(f"Transcription mode set to mode : {mmu.transcribe_mode}, language_config: {mmu.language_config}")

    # Checks if duplicate, if not updates the variables and return true
    duplicate_job = mmu.is_duplicate_job()

    if duplicate_job:
        mmu.fetch_data_from_s3()
        recording_duaration = mmu.audio_duration_in_seconds
        if transcription_format == 'vtt':
            transcript = mmu.output_vtt
        elif transcription_format == 'json':
            transcript = mmu.output_wljson
    else:
        # Downloading the video
        print("Starting video download.")
        mmu.download_video()
        print("Downloaded video file.")

        # Convert to WAV and split audio
        mmu.mp4_to_wav_convert()
        print("Converted mp4 to wav.")

        # Get audio duration
        recording_duaration = mmu.audio_duration()
        print(f"Recording duration : {recording_duaration}")

        if mmu.is_long_audio:
            transcript = None
        else:
            mmu.split_audio()
            print("Audio splitted")

            mmu.set_vocabulary(vocabulary_settings)
            completion_status = None
            try:
                # Transcription

                print("Starting concurrent transcription.")
                mmu.concurrent_transcription()
                print("Transcription complete!")

                print(
                    f"Converting {mmu.transcribe_mode} result transcription to standard format.")
                mmu.adapt_transcription()

                # Post transcription
                mmu.generate_vtt()
                print("VTT generation successful!")

                mmu.upload_transcripts_to_s3()
                print("Transcripts uploaded to S3 successfully!")

                mmu.upload_metadata_to_s3()
                print("Metadata uploaded to S3 successfully!")
                completion_status = 'SUCCESS'
            except Exception as e:
                completion_status = 'FAILED'
                print(str(e))
                print("An error occurred!", exc_info=True)
            finally:
                # Local and cloud cleanup
                print("Clearing local media directory.")
                mmu.clear_media_directory()
                print("Local media directory cleared!")
                print("Clearing S3 temporary directory.")
                mmu.clear_s3_temp_directory()
                print("S3 temporary directory cleared!")
                print(f"Task {completion_status} for bot_id {bot_id}")
                if completion_status == 'FAILED':
                    return "Task Failed!"
            if transcription_format == 'vtt':
                transcript = mmu.output_vtt
            elif transcription_format == 'json':
                transcript = mmu.output_wljson
            else:
                print('Invalid format, enter either vtt or json')
                return "Invalid Format!"
    return transcript, recording_duaration, mmu.is_long_audio


# Endpoint
@app.post("/meetings/transcribe")
def transcribe_meeting(input_data: InputData):
    '''

    '''
    bot_id = input_data.bot_id
    internal_language_code = input_data.internal_language_code
    transcription_format = input_data.transcription_format
    vocabulary_settings = input_data.vocabulary_settings
    transcription, recording_duration, is_long_audio = transcribe(
        bot_id, internal_language_code, transcription_format, vocabulary_settings)  # Stop flag will be active if content > 3 hours, transcription will not happen.
    return {
        "info":{
            "bot_id": bot_id,
            "internal_language_code": internal_language_code,
            "transcription_format": transcription_format,
            "is_long_audio": str(is_long_audio),
            "vocabulary_settings": vocabulary_settings
        },
        "results":{
            "transcription": transcription,
            "recording_duration": recording_duration}
        }

handler = Mangum(app)
