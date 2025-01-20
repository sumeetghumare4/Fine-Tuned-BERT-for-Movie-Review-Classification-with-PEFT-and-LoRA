# Fine-Tuned BERT for Movie Review Classification with PEFT and LoRA

This project implements a fine-tuned BERT model for movie review sentiment classification using Parameter Efficient Fine-Tuning (PEFT) and Low-Rank Adaptation (LoRA) techniques. The implementation is optimized for development in Visual Studio Code.

## Environment Setup in VS Code

1. Create a Python Virtual Environment:
   ```bash
   python -m venv venv
   ```

2. Activate the Virtual Environment:
   - On Windows:
     ```bash
     .\venv\Scripts\activate
     ```
   - On Linux/Mac:
     ```bash
     source venv/bin/activate
     ```

3. Install Recommended VS Code Extensions:
   - Python (`ms-python.python`)
   - Jupyter (`ms-toolsai.jupyter`)
   - Git (`ms-vscode.git`)

4. Select Python Interpreter in VS Code:
   - Press `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (Mac)
   - Type "Python: Select Interpreter"
   - Choose the interpreter from your virtual environment

## Installing Dependencies

Install all required packages using:
```bash
pip install -r requirements.txt
```

This will install all necessary dependencies including:
- datasets
- evaluate
- transformers[sentencepiece]
- accelerate
- peft

## Running the Code

### Using VS Code Debug Configuration

1. Open the project in VS Code
2. Navigate to the Debug view (`Ctrl+Shift+D` or `Cmd+Shift+D`)
3. Select "Python: Run fine_tuned_bert_for_movie_review_classification" from the debug configuration dropdown
4. Press F5 or click the green play button to start debugging

The debug configuration is already set up in `.vscode/launch.json` with the correct Python path and environment variables.

### Running from Terminal

Alternatively, you can run the script directly:
```bash
python src/fine_tuned_bert_for_movie_review_classification.py
```

## Project Structure

```
Fine-tuned/
├── src/
│   └── fine_tuned_bert_for_movie_review_classification.py
├── notebooks/
│   └── (Jupyter notebooks for experimentation)
├── tests/
│   └── (Test files)
├── .vscode/
│   └── launch.json
├── requirements.txt
└── README.md
```

- `src/`: Contains the main Python implementation files
- `notebooks/`: Jupyter notebooks for experimentation and analysis
- `tests/`: Unit tests and integration tests
- `.vscode/`: VS Code specific configurations
- `requirements.txt`: Project dependencies

## Contributing

1. Fork the repository
2. Create a new branch for your feature
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.