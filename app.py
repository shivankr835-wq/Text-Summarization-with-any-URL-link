
import validators
import streamlit as st

from youtube_transcript_api import YouTubeTranscriptApi

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from langchain_community.document_loaders import (
    UnstructuredURLLoader,
)


# STREAMLIT APP

st.set_page_config(
    page_title="LangChain: Summarize Text From YT or Website",
    page_icon="🦜"
)

st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader("Summarize URL")


# SIDEBAR - GROQ API KEY

with st.sidebar:
    groq_api_key = st.text_input(
        "Groq API Key",
        value="",
        type="password"
    )


# URL INPUT

generic_url = st.text_input(
    "URL",
    label_visibility="collapsed",
    placeholder="Enter YouTube or Website URL"
)


# PROMPT TEMPLATE

prompt_template = """
Provide a clear and concise summary of the following content
in approximately 300 words.

Content:
{text}
"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["text"]
)


# YOUTUBE TRANSCRIPT FUNCTION

def get_youtube_transcript(url):

    try:
        # Extract YouTube video ID

        if "youtu.be" in url:
            video_id = url.split("/")[-1].split("?")[0]

        elif "youtube.com" in url:
            video_id = url.split("v=")[1].split("&")[0]

        else:
            raise Exception("Invalid YouTube URL")

        # Create API object
        api = YouTubeTranscriptApi()

        # Get available transcripts
        transcript_list = api.list(video_id)

        # Get available language codes
        available_languages = [
            transcript.language_code
            for transcript in transcript_list
        ]

        # First preference: English
        if "en" in available_languages:
            selected_language = "en"

        # Second preference: Hindi
        elif "hi" in available_languages:
            selected_language = "hi"

        # Otherwise use first available language
        elif available_languages:
            selected_language = available_languages[0]

        else:
            raise Exception(
                "No transcript available for this video."
            )

        # Select transcript
        transcript = transcript_list.find_transcript(
            [selected_language]
        )

        # Fetch transcript
        transcript_data = transcript.fetch()

        # Convert transcript to text
        text = " ".join(
            item.text for item in transcript_data
        )

        return text

    except Exception as e:

        raise Exception(
            f"YouTube transcript error: {e}"
        )


# SUMMARIZE BUTTON

if st.button("Summarize the Content from YT or Website"):

    # Validate API Key and URL

    if not groq_api_key.strip():

        st.error("Please enter your Groq API Key.")

    elif not generic_url.strip():

        st.error("Please enter a URL.")

    elif not validators.url(generic_url):

        st.error(
            "Please enter a valid URL. "
            "It can be a YouTube video URL or website URL."
        )

    else:

        try:

            with st.spinner("Fetching and summarizing..."):

                # GROQ LLM

                llm = ChatGroq(
                    model="llama-3.1-8b-instant",
                    groq_api_key=groq_api_key
                )


                # LOAD YOUTUBE OR WEBSITE

                if (
                    "youtube.com" in generic_url
                    or "youtu.be" in generic_url
                ):

                    # YouTube transcript
                    text = get_youtube_transcript(
                        generic_url
                    )

                else:

                    # Website loader

                    loader = UnstructuredURLLoader(
                        urls=[generic_url],
                        ssl_verify=False,
                        headers={
                            "User-Agent": (
                                "Mozilla/5.0 "
                                "(Windows NT 10.0; Win64; x64) "
                                "AppleWebKit/537.36 "
                                "(KHTML, like Gecko) "
                                "Chrome/116.0.0.0 Safari/537.36"
                            )
                        }
                    )

                    # LOAD DOCUMENT

                    docs = loader.load()

                    if not docs:

                        st.error(
                            "Could not extract any content from the URL."
                        )

                    else:

                        # COMBINE DOCUMENT CONTENT

                        text = "\n\n".join(
                            doc.page_content
                            for doc in docs
                        )


                # CHECK TEXT

                if not text.strip():

                    st.error(
                        "Could not extract any content from the URL."
                    )

                else:

                    # LANGCHAIN LCEL CHAIN

                    chain = (
                        prompt
                        | llm
                        | StrOutputParser()
                    )


                    # GENERATE SUMMARY

                    output_summary = chain.invoke(
                        {
                            "text": text
                        }
                    )


                    # DISPLAY SUMMARY

                    st.success(
                        "Summary Generated Successfully!"
                    )

                    st.write(output_summary)


        except Exception as e:

            st.error(f"Exception: {e}")

