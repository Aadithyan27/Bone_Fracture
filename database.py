import sqlite3
from datetime import datetime
from typing import Optional, List, Dict, Any
import os

DATABASE_PATH = os.path.join(os.path.dirname(__file__), 'bone_fracture.db')

def get_connection():
    """Get database connection"""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row  # Enable column access by name
    return conn

def init_database():
    """Initialize database and create tables if they don't exist"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Create patients table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            gender TEXT NOT NULL,
            age INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create analysis_history table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id INTEGER,
            image_filename TEXT,
            prediction TEXT NOT NULL,
            confidence REAL NOT NULL,
            probabilities_healthy REAL,
            probabilities_fracture REAL,
            gradcam_image_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (patient_id) REFERENCES patients(id)
        )
    ''')
    
    conn.commit()
    conn.close()
    print(f"Database initialized at {DATABASE_PATH}")

# Patient operations
def create_patient(name: str, gender: str, age: int) -> int:
    """Create a new patient record and return patient ID"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute(
        'INSERT INTO patients (name, gender, age) VALUES (?, ?, ?)',
        (name, gender, age)
    )
    
    patient_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return patient_id

def get_patient(patient_id: int) -> Optional[Dict[str, Any]]:
    """Get patient by ID"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM patients WHERE id = ?', (patient_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return dict(row)
    return None

def get_all_patients() -> List[Dict[str, Any]]:
    """Get all patients"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM patients ORDER BY created_at DESC')
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]

# Analysis history operations
def save_analysis(
    patient_id: Optional[int],
    image_filename: str,
    prediction: str,
    confidence: float,
    probabilities_healthy: float,
    probabilities_fracture: float,
    gradcam_image_path: Optional[str] = None
) -> int:
    """Save analysis result and return analysis ID"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO analysis_history 
        (patient_id, image_filename, prediction, confidence, 
         probabilities_healthy, probabilities_fracture, gradcam_image_path)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (patient_id, image_filename, prediction, confidence,
          probabilities_healthy, probabilities_fracture, gradcam_image_path))
    
    analysis_id = cursor.lastrowid
    conn.commit()
    conn.close()
    
    return analysis_id

def get_analysis_history(limit: int = 50) -> List[Dict[str, Any]]:
    """Get recent analysis history with patient information"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT 
            a.*,
            p.name as patient_name,
            p.gender as patient_gender,
            p.age as patient_age
        FROM analysis_history a
        LEFT JOIN patients p ON a.patient_id = p.id
        ORDER BY a.created_at DESC
        LIMIT ?
    ''', (limit,))
    
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]

def get_patient_analyses(patient_id: int) -> List[Dict[str, Any]]:
    """Get all analyses for a specific patient"""
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT * FROM analysis_history
        WHERE patient_id = ?
        ORDER BY created_at DESC
    ''', (patient_id,))
    
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]

def get_analysis_stats() -> Dict[str, Any]:
    """Get statistics about analyses"""
    conn = get_connection()
    cursor = conn.cursor()
    
    # Total analyses
    cursor.execute('SELECT COUNT(*) as total FROM analysis_history')
    total = cursor.fetchone()['total']
    
    # Fracture count
    cursor.execute("SELECT COUNT(*) as count FROM analysis_history WHERE prediction = 'Fracture'")
    fracture_count = cursor.fetchone()['count']
    
    # Healthy count
    cursor.execute("SELECT COUNT(*) as count FROM analysis_history WHERE prediction = 'Healthy'")
    healthy_count = cursor.fetchone()['count']
    
    # Total patients
    cursor.execute('SELECT COUNT(*) as count FROM patients')
    patient_count = cursor.fetchone()['count']
    
    conn.close()
    
    return {
        'total_analyses': total,
        'fracture_count': fracture_count,
        'healthy_count': healthy_count,
        'total_patients': patient_count
    }

# Initialize database on module import
init_database()
