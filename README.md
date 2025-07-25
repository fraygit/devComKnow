# Developer Knowledgebase AI Chat

This project is a developer knowledgebase AI chat application. It is designed to assist developers by reading all markdown (`.md`) files from the repositories of a specified group in GitLab and using their content as context for answering questions.

## Features
- Connects to a GitLab group and accesses all repositories within that group.
- Reads and indexes all markdown documentation files from those repositories.
- Provides an AI-powered chat interface for developers to ask questions.
- Answers are generated using the context from the collected markdown files, making it easy to find and reference documentation across multiple projects.

## Usage
1. Configure the application with your GitLab group information and access credentials.
2. Start the application.
3. Use the chat interface to ask questions about your codebase and documentation.

## Requirements
- Python 3.12+
- See `requirements.txt` for dependencies.

## Purpose
This tool is intended to help development teams quickly find and utilize knowledge stored in markdown documentation across all their GitLab repositories, improving onboarding, productivity, and collaboration.
