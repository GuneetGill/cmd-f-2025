# EcoGPT

This Chrome extension is designed to reduce ChatGPT usage by promoting awareness of the environmental impact associated with AI models. The extension aims to help users make more conscious decisions about using AI-powered tools and reduce unnecessary usage, contributing to lowering the carbon footprint of these services.

## Features
**Usage Tracker**: Tracks the time spent interacting with ChatGPT and displays a usage summary.

**Environmental Impact Awareness**: Shows a message or alert when a user is about to interact with ChatGPT, informing them of the environmental impact of their actions.

**Alternative Suggestions**: When usage limits are approached, the extension suggests alternatives to ChatGPT for specific tasks, encouraging more eco-friendly choices such as Google search. 

## Technologies Used
JavaScript: Powers the frontend of the Chrome extension, handling both user interaction and core functionality.

Python: Serves as the backend engine, processing and analyzing environmental data and usage statistics.

GraphQL: The API is built with GraphQL, enabling flexible and efficient data retrieval

Docker: Utilized for containerizing the application, ensuring portability and seamless deployment across different environments.

Gemini AI: Leverages AI to assess user prompts, determining whether the request warrants ChatGPT’s assistance or if a simple Google search will suffice.

Render: Hosts and deploys the API to the cloud, providing reliable infrastructure and scaling.
Flask: A minimalistic Python web framework that powers the backend API and handles server-side requests.

Google Chrome: The Chrome extension is the primary interface, delivering an intuitive experience for users.

## Loading the extension

In Chrome Extension developer mode, add the `extensions` folder as unpacked.

## Authors:
Guneet Gill, Aniyah Bohnen, and Kaitlynn Chan

## docker

build
`docker build -t flask-api .`

run
`docker run -p 5000:5000 flask-api`
