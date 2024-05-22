# Qilo | Software and AI Engineer Intern Assignment

Welcome to the Qilo Software and AI Engineer Intern assignment repository! In this project, you will find the code implementation for the RAG (Retrieve, Aggregate, Generate) application as per the provided requirements.

## Overview

The RAG application is designed to perform the following features:

1. **Scrape "Luke Skywalker" Wiki Page**: The application scrapes the Wikipedia page of "Luke Skywalker" to gather relevant information.

2. **Chunking and Vector Database**: The scraped content is chunked and stored in a vector database like Faiss for efficient retrieval.

3. **Language Model API Calling**: The application integrates with a Language Model (LLM) API for generating answers to questions. You can choose 3 LLMs which are gpt2,bloom,openai-api.

4. **Question Answering System**: Users can ask questions through Postman or any other similar source. The application retrieves the top 3 relevant chunks related to the question, passes them along with the actual question to the LLM, and generates an answer.

## Features

### 1. Scraper
- The scraper module retrieves data from the Wikipedia page of "Luke Skywalker".
- It extracts relevant text content for further processing.

### 2. Chunking and Vector Database
- The chunking module segments the extracted text into smaller, meaningful chunks.
- These chunks are stored efficiently in a vector database like Faiss to enable fast retrieval.

### 3. Language Model API Integration
- Integration with a Language Model API is implemented for generating answers.
- You can easily swap out different LLM APIs based on requirements or availability.

### 4. Question Answering System
- Users can input questions through Postman or any similar tool.
- The system retrieves the most relevant chunks related to the question from the vector database.
- It then utilizes the Language Model API to generate accurate answers based on the question and relevant chunks.

## Q3. UI Implementation
- React is used for the ui part

## Getting Started

To get started with the project, follow these steps:

1. Clone the repository to your local machine.
2. Set up the necessary environment variables and dependencies as specified in the project documentation.
3. Run the application and test its functionality.
4. Explore and experiment with different LLM APIs and UI implementations to enhance the project further.

## Contact Information

For any inquiries or assistance regarding this project, please feel free to reach out to:

- Name: Savio babu
- Email: saviobabu007@gmail.com


