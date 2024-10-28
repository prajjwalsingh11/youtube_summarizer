import os
from openai import OpenAI
import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import yt_dlp
import tempfile
import re  # Add this at the top with other imports

# More flexible environment variable loading
def load_environment():
    """Load environment variables from .env file or system environment"""
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
    
    # Check for required environment variables
    required_vars = {
        'GROQ_API_KEY': "GROQ_API_KEY not found in environment variables",
        'YOUTUBE_EMAIL': "YOUTUBE_EMAIL not found in environment variables",
        'YOUTUBE_PASSWORD': "YOUTUBE_PASSWORD not found in environment variables"
    }
    
    missing_vars = []
    for var, message in required_vars.items():
        if not os.getenv(var):
            missing_vars.append(message)
    
    if missing_vars:
        raise ValueError("\n".join(missing_vars))
    
    return os.getenv('GROQ_API_KEY')

# Initialize clients with environment variables
try:
    api_key = load_environment()
    
    # Initialize Groq client with OpenAI compatibility
    groq_client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1"
    )

    # Separate client for Whisper
    whisper_client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1"
    )
except Exception as e:
    st.error(f"Error initializing API clients: {str(e)}")
    st.stop()

def download_audio(youtube_url):
    """Download audio from YouTube video with email/password authentication"""
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': '%(id)s.%(ext)s',
        'quiet': False,
        'verbose': True,
        # YouTube authentication using email and password
        'username': os.getenv('YOUTUBE_EMAIL'),
        'password': os.getenv('YOUTUBE_PASSWORD'),
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Referer': 'https://www.youtube.com/',
            'Origin': 'https://www.youtube.com',
        }
    }

    try:
        st.info("Starting YouTube authentication process...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Clear cache before starting
            ydl.cache.remove()
            
            try:
                # Try with email/password authentication
                info = ydl.extract_info(youtube_url, download=False)
                if info is None:
                    raise Exception("Could not extract video info")
                
                # Download with authentication
                ydl.download([youtube_url])
                video_id = info['id']
                audio_file = f"{video_id}.mp3"
                
                if os.path.exists(audio_file):
                    st.success(f"Audio downloaded successfully: {audio_file}")
                    return audio_file
                    
            except Exception as auth_error:
                st.warning(f"Authentication failed: {str(auth_error)}")
                st.info("Attempting alternative download method...")
                
                # Try alternative method
                alt_opts = ydl_opts.copy()
                alt_opts.update({
                    'format': 'bestaudio[ext=m4a]/bestaudio/best',
                    'username': None,
                    'password': None
                })
                
                with yt_dlp.YoutubeDL(alt_opts) as alt_ydl:
                    info = alt_ydl.extract_info(youtube_url, download=True)
                    audio_file = f"{info['id']}.mp3"
                    
                    if os.path.exists(audio_file):
                        st.success(f"Audio downloaded with alternative method: {audio_file}")
                        return audio_file
                    
        raise Exception("Could not download audio with any method")
                
    except Exception as e:
        st.error(f"All download attempts failed: {str(e)}")
        return None

def transcribe_audio(audio_file):
    """
    Transcribe audio using Groq's Whisper Large V3 Turbo
    """
    try:
        with open(audio_file, "rb") as audio:
            transcript = whisper_client.audio.transcriptions.create(
                model="whisper-large-v3-turbo",
                file=audio,
                response_format="text"
            )
        return transcript
    finally:
        if os.path.exists(audio_file):
            os.remove(audio_file)

def extract_video_id(youtube_url):
    """Extract video ID from different YouTube URL formats"""
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',  # Standard and shared URLs
        r'(?:embed\/)([0-9A-Za-z_-]{11})',   # Embed URLs
        r'(?:youtu\.be\/)([0-9A-Za-z_-]{11})',  # Shortened URLs
        r'(?:shorts\/)([0-9A-Za-z_-]{11})',   # YouTube Shorts
        r'^([0-9A-Za-z_-]{11})$'  # Just the video ID
    ]
    
    # Clean the URL first
    youtube_url = youtube_url.strip()
    
    for pattern in patterns:
        match = re.search(pattern, youtube_url)
        if match:
            return match.group(1)
    
    raise ValueError("Could not extract video ID from URL")

def get_transcript(youtube_url):
    """Get transcript from YouTube with enhanced error handling"""
    try:
        video_id = extract_video_id(youtube_url)
        st.info(f"Processing video ID: {video_id}")
        
        # First attempt: Try getting transcript directly
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            try:
                transcript = transcript_list.find_manually_created_transcript()
                st.success("Found manual transcript!")
            except:
                transcript = next(iter(transcript_list))
                st.success("Found auto-generated transcript!")
            
            full_transcript = " ".join([part['text'] for part in transcript.fetch()])
            language_code = transcript.language_code
            return full_transcript, language_code
            
        except Exception as e:
            st.warning(f"YouTube transcript not available: {str(e)}")
            
            # Audio transcription attempt
            st.info("Attempting to download and transcribe audio...")
            try:
                # Explicitly construct the full YouTube URL
                full_url = f"https://www.youtube.com/watch?v={video_id}"
                st.info(f"Downloading audio from: {full_url}")
                
                audio_file = download_audio(full_url)
                if audio_file and os.path.exists(audio_file):
                    st.success(f"Audio downloaded successfully: {audio_file}")
                    full_transcript = transcribe_audio(audio_file)
                    if full_transcript:
                        st.success("Audio transcription successful!")
                        return full_transcript, 'en'
                    else:
                        raise Exception("Transcription failed")
                else:
                    raise Exception("Audio download failed")
                    
            except Exception as e3:
                st.error(f"Audio processing failed: {str(e3)}")
                raise Exception(f"Could not process video: {str(e3)}")
        
    except Exception as e:
        st.error(f"Could not get transcript: {str(e)}")
        return None, None

def get_available_languages():
    """Return a dictionary of available languages"""
    return {
        'English': 'en',
        'Deutsch': 'de',
        'Español': 'es',
        'Français': 'fr',
        'Italiano': 'it',
        'Nederlands': 'nl',
        'Polski': 'pl',
        'Português': 'pt',
        '日本語': 'ja',
        '中文': 'zh',
        '한국어': 'ko',
        'Русский': 'ru'
    }

def create_summary_prompt(text, target_language):
    """Create an optimized prompt for summarization in the target language"""
    language_prompts = {
        'en': {
            'title': 'TITLE',
            'overview': 'OVERVIEW',
            'key_points': 'KEY POINTS',
            'takeaways': 'MAIN TAKEAWAYS',
            'context': 'CONTEXT & IMPLICATIONS'
        },
        'de': {
            'title': 'TITEL',
            'overview': 'ÜBERBLICK',
            'key_points': 'KERNPUNKTE',
            'takeaways': 'HAUPTERKENNTNISSE',
            'context': 'KONTEXT & AUSWIRKUNGEN'
        },
        # Add more languages as needed...
    }

    # Default to English if language not in dictionary
    prompts = language_prompts.get(target_language, language_prompts['en'])

    system_prompt = f"""You are an expert content analyst and summarizer. Create a comprehensive 
    summary in {target_language}. Ensure all content is fully translated and culturally adapted 
    to the target language."""

    user_prompt = f"""Please provide a detailed summary of the following content in {target_language}. 
    Structure your response as follows:

    🎯 {prompts['title']}: Create a descriptive title

    📝 {prompts['overview']} (2-3 sentences):
    - Provide a brief context and main purpose

    🔑 {prompts['key_points']}:
    - Extract and explain the main arguments
    - Include specific examples
    - Highlight unique perspectives

    💡 {prompts['takeaways']}:
    - List 3-5 practical insights
    - Explain their significance

    🔄 {prompts['context']}:
    - Broader context discussion
    - Future implications

    Text to summarize: {text}

    Ensure the summary is comprehensive enough for someone who hasn't seen the original content."""

    return system_prompt, user_prompt

def summarize_with_langchain_and_openai(transcript, language_code, model_name='llama-3.1-8b-instant'):
    # Split the document if it's too long
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len
    )
    texts = text_splitter.split_text(transcript)
    text_to_summarize = " ".join(texts[:4])  # Adjust this as needed

    system_prompt, user_prompt = create_summary_prompt(text_to_summarize, language_code)

    # Create summary using Groq's Llama model
    try:
        response = groq_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=4000  # Llama 3.2 1B has 8k token limit in preview
        )
        
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error with Groq API: {str(e)}")
        return None

def main():
    st.title('📺 Advanced YouTube Video Summarizer')
    st.markdown("""
    This tool creates comprehensive summaries of YouTube videos using advanced AI technology.
    It works with both videos that have transcripts and those that don't!
    """)
    
    # Create two columns for input fields
    col1, col2 = st.columns([3, 1])
    
    with col1:
        link = st.text_input('🔗 Enter YouTube video URL:')
    
    with col2:
        # Language selector
        languages = get_available_languages()
        target_language = st.selectbox(
            '🌍 Select Summary Language:',
            options=list(languages.keys()),
            index=0  # Default to English
        )
        # Convert display language to language code
        target_language_code = languages[target_language]

    if st.button('Generate Summary'):
        if link:
            try:
                with st.spinner('Processing...'):
                    progress = st.progress(0)
                    status_text = st.empty()

                    status_text.text('📥 Fetching video transcript...')
                    progress.progress(25)

                    transcript, _ = get_transcript(link)  # Original language doesn't matter now

                    status_text.text(f'🤖 Generating {target_language} summary...')
                    progress.progress(75)

                    summary = summarize_with_langchain_and_openai(
                        transcript, 
                        target_language_code,
                        model_name='llama-3.1-8b-instant'
                    )

                    status_text.text('✨ Summary Ready!')
                    st.markdown(summary)
                    progress.progress(100)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.warning('Please enter a valid YouTube link.')

if __name__ == "__main__":
    main()
