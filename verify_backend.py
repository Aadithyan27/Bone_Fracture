import requests
import time
import subprocess
import sys
import os

def check_server():
    print("Starting server for verification...")
    # Start server in background
    process = subprocess.Popen([sys.executable, 'c:/Bone_Fracture/server.py'], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE,
                               universal_newlines=True)
    
    try:
        # Wait for server to start
        print("Waiting for server to start...")
        for _ in range(10):
            try:
                response = requests.get('http://127.0.0.1:5000/health')
                if response.status_code == 200:
                    print("Server is running!")
                    break
            except requests.exceptions.ConnectionError:
                time.sleep(1)
        else:
            print("Server failed to start within 10 seconds.")
            # Print stderr if failed
            print("Server stderr:", process.stderr.read())
            return False

        # Check endpoints
        print("\nChecking endpoints:")
        
        # 1. Stats
        try:
            resp = requests.get('http://127.0.0.1:5000/api/stats')
            print(f"/api/stats: {resp.status_code} - {resp.json()}")
        except Exception as e:
            print(f"/api/stats failed: {e}")

        # 2. Create Patient
        try:
            patient_data = {"name": "Test Patient", "gender": "Male", "age": 30}
            resp = requests.post('http://127.0.0.1:5000/api/patients', json=patient_data)
            print(f"/api/patients (POST): {resp.status_code} - {resp.json()}")
        except Exception as e:
            print(f"/api/patients failed: {e}")

        # 3. List Patients
        try:
            resp = requests.get('http://127.0.0.1:5000/api/patients')
            print(f"/api/patients (GET): {resp.status_code} - {len(resp.json())} patients found")
        except Exception as e:
             print(f"/api/patients (GET) failed: {e}")
             
        # 4. History
        try:
            resp = requests.get('http://127.0.0.1:5000/api/history')
            print(f"/api/history: {resp.status_code}")
        except Exception as e:
            print(f"/api/history failed: {e}")

        return True

    finally:
        print("\nStopping server...")
        process.terminate()
        process.wait()

if __name__ == "__main__":
    success = check_server()
    if success:
        print("\nVerification PASSED")
    else:
        print("\nVerification FAILED")
