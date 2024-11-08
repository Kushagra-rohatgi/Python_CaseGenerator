from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from firebase_admin import credentials, initialize_app, auth, db
import requests
import os
import json
from dotenv import load_dotenv
import fitz

import spacy
import chromadb  # ChromaDB for local vector storage
from functools import wraps
import numpy as np
from typing import List, Dict, Any
import logging
from datetime import datetime
import re


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'server_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

logger.info("Starting server initialization...")

# Load environment variables
load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
firebase_db_url = os.getenv('FIREBASE_DATABASE_URL')

if not api_key:
    logger.error("GEMINI_API_KEY not found in environment variables")
    raise ValueError("GEMINI_API_KEY is required")

if not firebase_db_url:
    logger.error("FIREBASE_DATABASE_URL not found in environment variables")
    raise ValueError("FIREBASE_DATABASE_URL is required")

logger.info("Environment variables loaded successfully")

# Initialize Firebase Admin SDK
try:
    logger.info("Initializing Firebase...")
    firebase_config_path = './firebase-config/serviceAccountKey.json'
    
    if not os.path.exists(firebase_config_path):
        logger.error(f"Firebase config file not found at {firebase_config_path}")
        raise FileNotFoundError(f"Firebase config file not found at {firebase_config_path}")
    
    cred = credentials.Certificate(firebase_config_path)
    initialize_app(cred, {
        'databaseURL': firebase_db_url
    })
    logger.info("Firebase initialized successfully")
except Exception as e:
    logger.error(f"Error initializing Firebase: {str(e)}")
    raise

# Load spaCy model
try:
    logger.info("Loading spaCy model...")
    nlp = spacy.load("en_core_web_md")
    logger.info("spaCy model loaded successfully")
except OSError:
    logger.warning("spaCy model not found, downloading...")
    os.system("python -m spacy download en_core_web_md")
    nlp = spacy.load("en_core_web_md")
    logger.info("spaCy model downloaded and loaded")

# Initialize ChromaDB
try:
    logger.info("Initializing ChromaDB...")
    chroma_path = "./chroma_db"
    
    # Create directory if it doesn't exist
    if not os.path.exists(chroma_path):
        os.makedirs(chroma_path)
        logger.info(f"Created ChromaDB directory at {chroma_path}")
    
    client = chromadb.PersistentClient(path=chroma_path)
    
    try:
        collection = client.get_collection("case_study_vectors")
        logger.info("Retrieved existing ChromaDB collection")
    except:
        collection = client.create_collection("case_study_vectors")
        logger.info("Created new ChromaDB collection")
    
    logger.info("ChromaDB initialized successfully")
except Exception as e:
    logger.error(f"Error initializing ChromaDB: {str(e)}")
    raise

def parse_pdf(file_path: str) -> str:
    """Parse a PDF file and extract its text content."""
    logger.info(f"Attempting to parse PDF: {file_path}")
    try:
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        logger.info(f"Successfully parsed PDF, extracted {len(text)} characters")
        return text.strip()
    except Exception as e:
        logger.error(f"Error parsing PDF: {str(e)}")
        raise

def vectorize_text(text: str) -> np.ndarray:
    """Convert text to vector representation using spaCy."""
    logger.info(f"Vectorizing text of length: {len(text)}")
    try:
        doc = nlp(text)
        logger.info("Text vectorized successfully")
        return doc.vector
    except Exception as e:
        logger.error(f"Error vectorizing text: {str(e)}")
        raise

def store_vectors_in_chromadb(case_studies: List[Dict[str, Any]]) -> None:
    """Store text vectors in ChromaDB."""
    logger.info(f"Storing vectors for {len(case_studies)} case studies")
    try:
        for case_study in case_studies:
            text = case_study['text']
            vector = vectorize_text(text)
            vector_list = vector.tolist()
            
            collection.add(
                embeddings=[vector_list],
                documents=[text],
                ids=[str(case_study['id'])]
            )
        logger.info("Vectors stored successfully in ChromaDB")
    except Exception as e:
        logger.error(f"Error storing vectors in ChromaDB: {str(e)}")
        raise

def find_similar_case_study(query_vector: np.ndarray, n_results: int = 1) -> str:
    """Find similar case studies in ChromaDB."""
    logger.info("Searching for similar case studies")
    try:
        results = collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=n_results
        )
        
        if results and results['documents']:
            logger.info("Similar case study found")
            return results['documents'][0][0]
        logger.warning("No similar case study found")
        return ""
    except Exception as e:
        logger.error(f"Error finding similar case study: {str(e)}")
        raise

def generate_case_study(similar_case_study: str, parameters: Dict[str, Any]) -> str:
    """Generate a new case study using Gemini API."""
    logger.info("Generating case study with Gemini API")
    try:
        prompt = f"""
        Generate a product management case study based on the following reference and parameters:
        
        Reference case study:
        {similar_case_study}
        
        Parameters:
        {json.dumps(parameters, indent=2)}
        
        Please create a detailed case study that follows a similar structure but with unique content.
        Focus on {parameters.get('focus_area', 'Product Strategy')} with {parameters.get('difficulty', 'Medium')} difficulty.
        The case study should be relevant to the {parameters.get('industry', 'Technology')} industry.
        """
        
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key
        }
        
        payload = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }]
        }
        
        logger.info("Sending request to Gemini API...")
        response = requests.post(url, headers=headers, json=payload)
        logger.info(f"Gemini API response status: {response.status_code}")
        
        if response.status_code == 200:
            content = response.json()
            generated_text = content['candidates'][0]['content']['parts'][0]['text']
            logger.info(f"Successfully generated case study of length: {len(generated_text)}")
            return generated_text
        else:
            logger.error(f"Gemini API error: {response.text}")
            raise Exception(f"Failed to generate case study: {response.text}")
    except Exception as e:
        logger.error(f"Error in generate_case_study: {str(e)}")
        raise

def generate_qa(case_study: str, parameters: Dict[str, Any]) -> List[Dict[str, str]]:
    """Generate questions and answers using Gemini API."""
    logger.info("Generating Q&A with Gemini API")
    try:
        prompt = f"""
        Based on this case study, generate 5 questions and answers in the following JSON format:
        [
            {{
                "question": "First question here?",
                "answer": "Detailed answer here"
            }},
            // ... more Q&A pairs
        ]

        Case Study:
        {case_study}

        Requirements:
        - Generate exactly 5 questions
        - Questions should be at {parameters.get('difficulty', 'Medium')} difficulty level
        - Focus on {parameters.get('focus_area', 'Product Strategy')}
        - Each answer should be detailed and thorough
        - Strictly maintain the JSON format specified above
        """
        
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key
        }
        
        payload = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }]
        }
        
        logger.info("Sending request to Gemini API for Q&A...")
        response = requests.post(url, headers=headers, json=payload)
        logger.info(f"Gemini API Q&A response status: {response.status_code}")
        
        if response.status_code == 200:
            content = response.json()
            qa_text = content['candidates'][0]['content']['parts'][0]['text']
            logger.info(f"Raw Q&A response: {qa_text}")
            
            try:
                # Try to find JSON array in the response
                json_match = re.search(r'\[.*\]', qa_text.replace('\n', ''), re.DOTALL)
                if json_match:
                    qa_text = json_match.group()
                
                qa_list = json.loads(qa_text)
                logger.info(f"Successfully generated {len(qa_list)} Q&A pairs")
                return qa_list
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing Q&A JSON response: {str(e)}")
                logger.info("Falling back to manual Q&A extraction")
                return extract_qa_pairs(qa_text)
        else:
            logger.error(f"Gemini API Q&A error: {response.text}")
            return [{
                "question": "Default question about the case study?",
                "answer": "Due to an error, we couldn't generate specific questions. Please try again."
            }]
    except Exception as e:
        logger.error(f"Error in generate_qa: {str(e)}")
        return [{
            "question": "Default question about the case study?",
            "answer": f"An error occurred while generating questions: {str(e)}"
        }]

def extract_qa_pairs(text: str) -> List[Dict[str, str]]:
    """Extract Q&A pairs from text if JSON parsing fails."""
    logger.info("Attempting to extract Q&A pairs from text")
    qa_pairs = []
    lines = text.split('\n')
    current_question = None
    current_answer = []
    
    for line in lines:
        line = line.strip()
        if line.startswith(('Q:', 'Question:')):
            if current_question:
                qa_pairs.append({
                    'question': current_question,
                    'answer': ' '.join(current_answer)
                })
            current_question = line.split(':', 1)[1].strip()
            current_answer = []
        elif line.startswith(('A:', 'Answer:')) and current_question:
            answer_text = line.split(':', 1)[1].strip()
            current_answer.append(answer_text)
        elif line and current_question:
            current_answer.append(line)
    
    if current_question:
        qa_pairs.append({
            'question': current_question,
            'answer': ' '.join(current_answer)
        })
    
    logger.info(f"Extracted {len(qa_pairs)} Q&A pairs")
    return qa_pairs

@app.route('/')
def index():
    """Serve the HTML interface."""
    return send_file('test.html')

@app.route('/test', methods=['GET'])
def test():
    """Test endpoint to verify server is working."""
    logger.info("Test endpoint called")
    return jsonify({
        "message": "Server is working!",
        "status": "OK",
        "endpoints": {
            "test": "GET /test",
            "generate": "POST /generate"
        },
        "api_status": {
            "firebase": "connected" if firebase_db_url else "not configured",
            "gemini": "configured" if api_key else "not configured",
            "spacy": "loaded",
            "chromadb": "initialized"
        }
    })


@app.route('/generate', methods=['POST'])
def generate():
    """Generate endpoint for case study generation."""
    logger.info("\n=== Starting generate endpoint ===")
    
    try:
        # Check if the request has JSON content
        if not request.is_json:
            logger.error("Request must be JSON")
            return jsonify({"error": "Request must be JSON"}), 400
        
        logger.info(f"Request received: {request.json}")
        
        parameters = request.json.get('parameters')
        logger.info(f"Parameters received: {parameters}")
        
        if not parameters:
            logger.error("No parameters provided")
            return jsonify({"error": "No parameters provided"}), 400

        # Validate required parameters
        required_params = ['industry', 'role', 'difficulty', 'focus_area']
        missing_params = [param for param in required_params if param not in parameters]
        if missing_params:
            logger.error(f"Missing required parameters: {missing_params}")
            return jsonify({"error": f"Missing required parameters: {missing_params}"}), 400

        # Check or create the data directory
        data_dir = "./Data"
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            logger.info(f"Created Data directory at {data_dir}")
        
        # Initialize case study files and parse PDFs
        case_study_files = ["./Data/case_study1.pdf"]
        case_studies = []
        
        for i, file_path in enumerate(case_study_files):
            logger.info(f"Processing file {i+1}/{len(case_study_files)}: {file_path}")
            
            if not os.path.exists(file_path):
                logger.error(f"Case study file not found at {file_path}")
                return jsonify({"error": f"Case study file not found: {file_path}"}), 404
            
            text = parse_pdf(file_path)
            if not text:
                logger.error("No text extracted from PDF")
                return jsonify({"error": "Failed to extract text from PDF"}), 500
            
            case_studies.append({"id": i, "text": text})
            logger.info(f"Successfully processed file {i+1}")

        # Store vectors and find a similar case study
        logger.info("Storing vectors in ChromaDB...")
        store_vectors_in_chromadb(case_studies)
        
        logger.info("Finding similar case study...")
        query_vector = vectorize_text("Generate a product management case study")
        similar_case_study = find_similar_case_study(query_vector)
        
        if not similar_case_study:
            logger.error("No similar case study found")
            return jsonify({"error": "Failed to find similar case study"}), 500

        # Generate new case study and Q&A pairs
        logger.info("Generating new case study...")
        case_study = generate_case_study(similar_case_study, parameters)
        
        logger.info("Generating Q&A...")
        qa = generate_qa(case_study, parameters)
        logger.info(f"Q&A generation completed. Generated {len(qa)} pairs")

        # Prepare response data
        response_data = {
            "case_study": case_study,
            "questions_and_answers": qa,
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "parameters": parameters
            }
        }

        logger.info("=== Generate endpoint completed successfully ===\n")
        return jsonify(response_data)
    
    except Exception as e:
        logger.error(f"Error in generate endpoint: {str(e)}")
        return jsonify({
            "error": str(e),
            "case_study": "Error generating case study",
            "questions_and_answers": [{
                "question": "Error occurred",
                "answer": f"An error occurred while generating the content: {str(e)}"
            }]
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)