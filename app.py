# import os 
# import json
# from dotenv import load_dotenv
# from flask import Flask,request, jsonify
# from langchain_community.document_loaders import WebBaseLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_groq import ChatGroq
# import re
# from typing import Dict, List



# load_dotenv()
# GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")
# GROQ_API_KEY=os.getenv("GROQ_API_KEY")

# def string_to_json(text):
#     try:
#         match = re.search(r'\{.*\}', text, re.DOTALL)
#         if not match:
#             raise ValueError("No JSON object found in the string.")
        
#         json_str = match.group(0).strip()
#         return json.loads(json_str)
#     except (json.JSONDecodeError, ValueError) as e:
#         print(f"Error parsing JSON: {e}")
#         return None


# splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200)
# embedding_model=GoogleGenerativeAI(model="models/embedding-001", api_key=GOOGLE_API_KEY)
# llm=ChatGroq(model="LLaMA3-8b-8192",temperature=0, api_key=GROQ_API_KEY)

# prompt=ChatPromptTemplate.from_messages([
# """
# You are a useful Website  analyzer

# You will be provided with 
# 1. Website content.
# 2. Topics to explore.

# You will perform the following tasks:
# 1. Analyze the content and provide a summary of the key points also with its sentiment score.
# 2. Provide a list of key topics discussed in the content with relevant to topics.
# 3. Create a roadmap for further exploration of the topics.
# 4. Provide a list of potential questions that could be asked based on the content.
# 5. Provide the response in JSON format with the following keys strictly no extra keys:
# - summary: Summary of the key points with sentiment score.
# - key_topics: List of key topics discussed in the content.
# - roadmap: Roadmap for further exploration of the topics.
# - questions: List of potential questions that could be asked based on the content.
# - sentiment_score: Sentiment score of the content.

# --- Input ---
# {content}
# {topics}
# --- Output ---
# {{"summary": "Provide a summary of the key points with sentiment score"}},
# {{"key_topics": "List of key topics discussed in the content"}},
# {{"roadmap": "Roadmap for further exploration of the topics"}},
# {{"questions": "List of potential questions that could be asked based on the content"}}
# {{"sentiment_score": "Sentiment score of the content"}}

# Follow this format strictly dont provide any extra headers and details
# """
# ])

# def analyze_website():
#     data= request.get_json()
#     urls = data.get("urls",[])
#     topics= data.get("topics", [])
#     topics = ", ".join(topics) if topics else ""
#     content=[]
#     for url in urls:
#         if not url.startswith("http"):
#             return jsonify({"status": "error", "message": "Invalid URL format"}), 400
#         loader = WebBaseLoader(url)
#         docs = loader.load()
#         content.extend(docs)
#     splits = splitter.split_documents(docs)
#     formatted_prompt = prompt.format_prompt(content=splits, topics=topics)
#     response = llm.invoke(formatted_prompt)
#     parsed_response = string_to_json(response.content)
#     return jsonify({
#         "status": "success", 
#         "message": "Website analysis completed successfully",
#         "data": {
#             "summary": parsed_response.get("summary", ""),
#             "sentiment_score": parsed_response.get("sentiment_score", ""),
#             "key_topics": parsed_response.get("key_topics", []),
#             "roadmap": parsed_response.get("roadmap", []),
#             "questions": parsed_response.get("questions", [])
#         }
#     }), 200


import os
import json
import re
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# Load API keys from .env
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Function to safely parse JSON from model response
def string_to_json(text):
    try:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in the string.")
        json_str = match.group(0).strip()
        return json.loads(json_str)
    except (json.JSONDecodeError, ValueError) as e:
        st.error(f"Error parsing JSON: {e}")
        return None

# LangChain setup
splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=200)
embedding_model = GoogleGenerativeAI(model="models/embedding-001", api_key=GOOGLE_API_KEY)
llm = ChatGroq(model="LLaMA3-8b-8192", temperature=0, api_key=GROQ_API_KEY)

prompt = ChatPromptTemplate.from_messages([
"""
You are a useful Website analyzer

You will be provided with:
1. Website content.
2. Topics to explore.

You will perform the following tasks:
1. Analyze the content and provide a summary of the key points also with its sentiment score.
2. Provide a list of key topics discussed in the content with relevant to topics.
3. Create a roadmap for further exploration of the topics.
4. Provide a list of potential questions that could be asked based on the content.
5. Provide the response in JSON format with the following keys strictly no extra keys:
- summary
- key_topics
- roadmap
- questions
- sentiment_score

--- Input ---
{content}
{topics}
--- Output ---
{{"summary": "Provide a summary of the key points with sentiment score"}},
{{"key_topics": "List of key topics discussed in the content"}},
{{"roadmap": "Roadmap for further exploration of the topics"}},
{{"questions": "List of potential questions that could be asked based on the content"}},
{{"sentiment_score": "Sentiment score of the content"}}

Follow this format strictly; no extra headers or details.
"""
])

port = os.environ.get("PORT")
st.set_page_config(page_title="NASA Web Analyzer", layout="wide")

# Streamlit UI
st.title("🌐 Website Analyzer")

urls_input = st.text_area("Enter website URLs (one per line):")
topics_input = st.text_area("Enter topics to explore (comma-separated):")

if st.button("Analyze Website"):
    urls = [url.strip() for url in urls_input.split("\n") if url.strip()]
    topics = ", ".join([t.strip() for t in topics_input.split(",") if t.strip()])

    if not urls:
        st.error("Please enter at least one valid URL.")
    else:
        try:
            content = []
            for url in urls:
                if not url.startswith("http"):
                    st.error(f"Invalid URL format: {url}")
                    st.stop()
                loader = WebBaseLoader(url)
                docs = loader.load()
                content.extend(docs)

            splits = splitter.split_documents(content)
            formatted_prompt = prompt.format_prompt(content=splits, topics=topics)
            response = llm.invoke(formatted_prompt)

            parsed_response = string_to_json(response.content)
            if parsed_response:
                st.success("✅ Website analysis completed successfully")

                # Summary
                st.subheader("📄 Summary")
                st.markdown(f"**{parsed_response.get('summary', 'No summary available.')}**")

                # Sentiment Score
                st.subheader("📊 Sentiment Score")
                st.progress(float(parsed_response.get("sentiment_score", 0)))
                st.write(f"**Score:** {parsed_response.get('sentiment_score', 0)}")

                # Key Topics
                st.subheader("🗂 Key Topics")
                key_topics = parsed_response.get("key_topics", [])
                if key_topics:
                    for i, topic in enumerate(key_topics, 1):
                        st.markdown(f"- {topic}")
                else:
                    st.write("No key topics found.")

                # Roadmap
                st.subheader("🛣 Roadmap")
                roadmap = parsed_response.get("roadmap", [])
                if roadmap:
                    for i, step in enumerate(roadmap, 1):
                        st.markdown(f"{i}. {step}")
                else:
                    st.write("No roadmap available.")

                # Questions
                st.subheader("❓ Questions")
                questions = parsed_response.get("questions", [])
                if questions:
                    for i, q in enumerate(questions, 1):
                        st.markdown(f"{i}. {q}")
                else:
                    st.write("No questions generated.")

        except Exception as e:
            st.error(f"An error occurred: {e}")
