# MedBot

MedBot is an AI-powered medical assistant designed to help users with medical queries, information retrieval, and healthcare guidance. It leverages vector search and AI models to provide relevant, accurate, and timely responses.

# MedBot

## Demo Video

[Watch the demo video on Google Drive](https://drive.google.com/file/d/17_sp5uO_WfJ2pCOP3qJgCzAIgC_qjtxN/view?usp=sharing)

## Features

- **Medical Query Handling:** Ask questions about symptoms, conditions, medications, and more.
- **Vector Search:** Uses FAISS-based vectorstore for efficient semantic search.
- **AI Integration:** Integrates with language models for natural language understanding and response generation.
- **Extensible:** Easily add new data sources or expand capabilities.

## Project Structure

```
MedBot/
│
├── vectorstore/
│   └── index.faiss         # FAISS vector index for semantic search
├── data/                   # Medical data sources (if any)
├── src/                    # Source code for MedBot
│   ├── ...                 # Python/JS/other source files
├── tests/                  # Unit and integration tests
├── requirements.txt        # Python dependencies (if applicable)
├── package.json            # Node.js dependencies (if applicable)
└── README.md               # Project documentation
```

## Getting Started

1. **Clone the repository:**
   ```sh
   git clone https://github.com/Yashwanth353/MedBot.git
   cd MedBot
   ```

2. **Install dependencies:**
   - For Python:
     ```sh
     pip install -r requirements.txt
     ```
   - For Node.js:
     ```sh
     npm install
     ```

3. **Run the application:**
   - For Python:
     ```sh
     python src/main.py
     ```
   - For Node.js:
     ```sh
     npm start
     ```

4. **Interact with MedBot:**
   - Use the provided CLI, web interface, or API (depending on implementation).

## Usage

- Ask medical questions in natural language.
- Retrieve information about diseases, symptoms, and treatments.
- Get suggestions for further reading or next steps.

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements, bug fixes, or new features.

## License

This project is licensed under the MIT License.

## Disclaimer

MedBot is for informational purposes only and does not provide medical advice. Always consult a healthcare professional for medical