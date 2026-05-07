# LAB - NormalObjects - Creative Complaint Handler (LangChain)

## 📋 Project Overview

This project implements a **Creative Complaint Handler** using LangChain and LangGraph. It processes user complaints through different creative personas (Philosopher, Comedian, Therapist, etc.) powered by OpenAI's GPT-4o-mini model, offering absurdist, humorous, and philosophical perspectives on everyday problems.

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key
- pip (Python package manager)

### Installation

1. **Clone the repository** (if not already cloned):
   ```bash
   git clone <repository-url>
   cd "LAB | NormalObjects - Creative Complaint Handler (LangChain)"
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   
   Key dependencies:
   - `langchain`
   - `langchain-openai`
   - `langgraph`
   - `python-dotenv`

4. **Set up environment variables**:
   - Create a `.env` file in the project root
   - Add your OpenAI API key:
     ```
     OPENAI_API_KEY=your_api_key_here
     ```

### Running the Application

**Basic execution**:
```bash
python normalobjects_langchain.py
```

**With custom input**:
```bash
python normalobjects_langchain.py --complaint "Your complaint here"
```

**Run analysis script**:
```bash
python step5_analysis.py
```

## 📁 Project Structure

```
.
├── normalobjects_langchain.py    # Main application with LangChain agent and tools
├── step5_analysis.py             # Analysis and evaluation script
├── analysis_report.md            # Generated analysis report
├── output normalobjects_langchain.md  # Output and results documentation
├── .env                          # Environment variables (API keys)
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

## 🛠️ Working Instructions

### Step 1: Environment Setup

- Ensure `.env` file is configured with a valid `OPENAI_API_KEY`
- The `.env` file is excluded from version control for security
- Never commit API keys or sensitive credentials

### Step 2: Running the Main Application

The `normalobjects_langchain.py` script:
1. Initializes the LangChain environment and loads environment variables
2. Creates specialized tools (consult_philosopher, consult_comedian, consult_therapist, etc.)
3. Sets up a ReAct agent using LangGraph
4. Processes user complaints through multiple creative personas
5. Outputs creative responses from different perspectives

**Example workflow**:
```bash
# Run the main application
python normalobjects_langchain.py

# Enter a complaint when prompted
# Wait for the agent to process through all personas
# Review the creative responses
```

### Step 3: Analyzing Results

Use `step5_analysis.py` to:
- Evaluate the quality of responses
- Analyze patterns in complaint handling
- Generate performance metrics
- Create analysis reports in Markdown format

```bash
python step5_analysis.py
```

### Step 4: Review Output

- Check `analysis_report.md` for detailed analysis
- Check `output normalobjects_langchain.md` for application output
- Review console output for execution logs and debugging information

## 🧠 Key Components

### Main Tools

1. **consult_philosopher**: Provides existential and philosophical perspectives
2. **consult_comedian**: Offers absurdist, humorous takes on complaints
3. **consult_therapist**: Delivers empathetic, therapeutic viewpoints
4. **consult_absurdist**: Embraces the absurdity of the situation
5. Additional creative tools as needed

### Agent Architecture

- Uses **LangGraph's ReAct agent** for structured problem-solving
- Leverages **OpenAI's GPT-4o-mini** for natural language processing
- Implements **callback handlers** for logging and tracking

## 🔧 Troubleshooting

### Common Issues

**"OPENAI_API_KEY not found"**
- Ensure `.env` file exists in project root
- Verify the API key is correctly set
- Check that `python-dotenv` is installed

**Module import errors**
- Install all dependencies: `pip install -r requirements.txt`
- Ensure virtual environment is activated
- Check Python version compatibility (3.8+)

**Rate limiting from OpenAI**
- Check your API quota and usage
- Implement request delays if needed
- Review OpenAI documentation for rate limits

### Debug Mode

To enable detailed logging:
```bash
# Set environment variable for verbose output
export LANGCHAIN_DEBUG=true
python normalobjects_langchain.py
```

## 📊 Output Files

- **analysis_report.md**: Detailed analysis of responses and metrics
- **output normalobjects_langchain.md**: Full application output and results
- **Console output**: Real-time execution logs and status messages

## 🔒 Security Best Practices

- ✅ Keep `.env` file in `.gitignore` (already configured)
- ✅ Never commit API keys to version control
- ✅ Use environment variables for sensitive data
- ✅ Regularly rotate API keys
- ✅ Monitor API usage for unauthorized access

## 📝 Development Notes

- The application uses OpenAI's GPT-4o-mini model with `temperature=0.7` for creative responses
- All tools are implemented as LangChain tools using the `@tool` decorator
- The ReAct agent automatically selects appropriate tools based on the complaint
- Output includes creative personas' responses with visual indicators (emojis)

## 📚 Additional Resources

- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Python dotenv Guide](https://github.com/theskumar/python-dotenv)