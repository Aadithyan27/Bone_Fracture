import { useState, useRef, useEffect } from 'react';
import './index.css';

interface Patient {
    id: number;
    name: string;
    gender: string;
    age: number;
}

interface AnalysisResult {
    prediction: string;
    confidence: string;
    probabilities: {
        Healthy: string;
        Fracture: string;
    };
    gradcam_image: string;
}

interface AnalysisHistoryItem {
    id: number;
    patient_name: string;
    patient_age: number;
    patient_gender: string;
    image_filename: string;
    prediction: string;
    confidence: number;
    gradcam_image_path: string;
    created_at: string;
}

function App() {
    const [view, setView] = useState<'home' | 'history'>('home');
    const [patient, setPatient] = useState<Patient | null>(null);

    return (
        <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
            <nav style={{ background: 'white', borderBottom: '1px solid var(--border-color)', padding: '1rem 2rem' }}>
                <div className="container" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: 0 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <div style={{ width: '32px', height: '32px', background: 'var(--primary-color)', borderRadius: '6px' }}></div>
                        <span className="title-medical" style={{ fontSize: '1.25rem' }}>Bone Fracture Detection AI </span>
                    </div>
                    <div style={{ display: 'flex', gap: '1rem' }}>
                        <button
                            className={`btn ${view === 'home' ? 'btn-primary' : 'btn-secondary'}`}
                            onClick={() => setView('home')}
                        >
                            New Analysis
                        </button>
                        <button
                            className={`btn ${view === 'history' ? 'btn-primary' : 'btn-secondary'}`}
                            onClick={() => setView('history')}
                        >
                            Patient History
                        </button>
                    </div>
                </div>
            </nav>

            <main className="container" style={{ flex: 1, marginTop: '2rem' }}>
                {view === 'home' ? (
                    !patient ? (
                        <PatientForm onPatientSubmit={setPatient} />
                    ) : (
                        <AnalysisView patient={patient} onReset={() => setPatient(null)} />
                    )
                ) : (
                    <HistoryView />
                )}
            </main>

            <footer style={{ textAlign: 'center', padding: '2rem', color: 'var(--text-dim)', fontSize: '0.875rem' }}>
                &copy; 2026 Bone Fracture Detection AI System. For Investigational Use Only.
            </footer>
        </div>
    );
}

function PatientForm({ onPatientSubmit }: { onPatientSubmit: (p: Patient) => void }) {
    const [formData, setFormData] = useState({ name: '', gender: 'Male', age: '' });
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setLoading(true);
        setError(null);

        try {
            const response = await fetch('/api/patients', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(formData)
            });

            if (!response.ok) throw new Error('Failed to register patient');

            const data = await response.json();
            onPatientSubmit({
                id: data.id,
                name: formData.name,
                gender: formData.gender,
                age: parseInt(formData.age)
            });
        } catch (err) {
            setError('Error registering patient. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="card animate-fade-in" style={{ maxWidth: '600px', margin: '2rem auto' }}>
            <h2 className="title-medical" style={{ marginBottom: '1.5rem', fontSize: '1.5rem' }}>Patient Registration</h2>
            <form onSubmit={handleSubmit}>
                <div className="form-group">
                    <label className="form-label">Patient Name</label>
                    <input
                        className="form-input"
                        required
                        value={formData.name}
                        onChange={e => setFormData({ ...formData, name: e.target.value })}
                        placeholder="Enter full name"
                    />
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                    <div className="form-group">
                        <label className="form-label">Gender</label>
                        <select
                            className="form-select"
                            value={formData.gender}
                            onChange={e => setFormData({ ...formData, gender: e.target.value })}
                        >
                            <option value="Male">Male</option>
                            <option value="Female">Female</option>
                            <option value="Other">Other</option>
                        </select>
                    </div>
                    <div className="form-group">
                        <label className="form-label">Age</label>
                        <input
                            type="number"
                            className="form-input"
                            required
                            min="0"
                            max="120"
                            value={formData.age}
                            onChange={e => setFormData({ ...formData, age: e.target.value })}
                        />
                    </div>
                </div>

                {error && <div style={{ color: 'var(--danger)', marginBottom: '1rem' }}>{error}</div>}

                <button type="submit" className="btn btn-primary" style={{ width: '100%' }} disabled={loading}>
                    {loading ? 'Registering...' : 'Start Examination'}
                </button>
            </form>
        </div>
    );
}

function AnalysisView({ patient, onReset }: { patient: Patient, onReset: () => void }) {
    const [file, setFile] = useState<File | null>(null);
    const [preview, setPreview] = useState<string | null>(null);
    const [result, setResult] = useState<AnalysisResult | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files && e.target.files[0]) {
            const selectedInfo = e.target.files[0];
            setFile(selectedInfo);
            setPreview(URL.createObjectURL(selectedInfo));
            setResult(null);
            setError(null);
        }
    };

    const handleAnalyze = async () => {
        if (!file) return;
        setLoading(true);
        setError(null);

        const formData = new FormData();
        formData.append('file', file);
        formData.append('patient_id', patient.id.toString());

        try {
            const response = await fetch('/api/analyze', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) throw new Error('Analysis failed');
            const data = await response.json();
            setResult(data);
        } catch (err) {
            setError('Failed to analyze image. Ensure backend is running.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="animate-fade-in">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
                <h2 className="title-medical" style={{ margin: 0, fontSize: '1.5rem' }}>X-Ray Analysis</h2>
                <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
                    <span style={{ color: 'var(--text-dim)' }}>
                        Patient: <strong>{patient.name}</strong> ({patient.gender}, {patient.age})
                    </span>
                    <button className="btn btn-secondary" style={{ padding: '0.25rem 0.75rem', fontSize: '0.875rem' }} onClick={onReset}>
                        Change Patient
                    </button>
                </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: result ? '1fr 1fr' : '1fr', gap: '2rem', transition: 'all 0.5s ease' }}>
                <div className="card">
                    <h3 style={{ marginTop: 0, color: 'var(--text-dim)', fontSize: '1rem' }}>X-Ray Upload</h3>
                    <div
                        style={{
                            border: '2px dashed var(--border-color)',
                            borderRadius: 'var(--radius)',
                            padding: '2rem',
                            textAlign: 'center',
                            cursor: 'pointer',
                            background: '#f8fafc',
                            marginBottom: '1.5rem',
                            minHeight: '300px',
                            display: 'flex',
                            flexDirection: 'column',
                            justifyContent: 'center',
                            alignItems: 'center'
                        }}
                        onClick={() => fileInputRef.current?.click()}
                    >
                        <input
                            type="file"
                            ref={fileInputRef}
                            onChange={handleFileChange}
                            accept="image/*"
                            style={{ display: 'none' }}
                        />

                        {preview ? (
                            <img src={preview} alt="Preview" style={{ maxHeight: '280px', maxWidth: '100%', borderRadius: '4px' }} />
                        ) : (
                            <div style={{ color: 'var(--text-dim)' }}>
                                <div style={{ fontSize: '2rem', marginBottom: '0.5rem' }}>📷</div>
                                <p>Click or drag X-ray image here</p>
                            </div>
                        )}
                    </div>

                    <button
                        className="btn btn-primary"
                        style={{ width: '100%' }}
                        onClick={handleAnalyze}
                        disabled={!file || loading}
                    >
                        {loading ? 'Analyzing Image...' : 'Run Diagnostic Analysis'}
                    </button>
                    {error && <p style={{ color: 'var(--danger)', marginTop: '1rem', textAlign: 'center' }}>{error}</p>}
                </div>

                {result && (
                    <div className="card animate-fade-in" style={{ animationDelay: '0.2s' }}>
                        <h3 style={{ marginTop: 0, borderBottom: '1px solid var(--border-color)', paddingBottom: '1rem', marginBottom: '1.5rem' }}>Diagnostic Results</h3>

                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '2rem' }}>
                            <div style={{ padding: '1rem', background: '#f8fafc', borderRadius: '8px', textAlign: 'center' }}>
                                <div style={{ fontSize: '0.875rem', color: 'var(--text-dim)', marginBottom: '0.25rem' }}>Prediction</div>
                                <div style={{
                                    fontSize: '1.5rem',
                                    fontWeight: 'bold',
                                    color: result.prediction === 'Fracture' ? 'var(--danger)' : 'var(--success)'
                                }}>
                                    {result.prediction.toUpperCase()}
                                </div>
                            </div>
                            <div style={{ padding: '1rem', background: '#f8fafc', borderRadius: '8px', textAlign: 'center' }}>
                                <div style={{ fontSize: '0.875rem', color: 'var(--text-dim)', marginBottom: '0.25rem' }}>Confidence</div>
                                <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: 'var(--primary-dark)' }}>
                                    {result.confidence}
                                </div>
                            </div>
                        </div>

                        <div style={{ marginBottom: '1.5rem' }}>
                            <div style={{ fontSize: '0.875rem', fontWeight: 600, marginBottom: '0.5rem' }}>Grad-CAM Analysis</div>
                            <div style={{ border: '1px solid var(--border-color)', borderRadius: '8px', overflow: 'hidden' }}>
                                <img src={result.gradcam_image} alt="Grad-CAM" style={{ width: '100%', display: 'block' }} />
                            </div>
                        </div>

                        <div style={{ fontSize: '0.875rem', color: 'var(--text-dim)' }}>
                            Detailed Probabilities: <br />
                            Healthy: <strong>{result.probabilities.Healthy}</strong> •
                            Fracture: <strong>{result.probabilities.Fracture}</strong>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}

function HistoryView() {
    const [history, setHistory] = useState<AnalysisHistoryItem[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetch('/api/history')
            .then(res => res.json())
            .then(data => setHistory(data))
            .catch(console.error)
            .finally(() => setLoading(false));
    }, []);

    if (loading) return <div className="text-center" style={{ padding: '4rem' }}>Loading records...</div>;

    return (
        <div className="card animate-fade-in">
            <h2 className="title-medical" style={{ fontSize: '1.5rem', marginBottom: '1.5rem' }}>Patient History</h2>
            {history.length === 0 ? (
                <p style={{ color: 'var(--text-dim)', textAlign: 'center', padding: '2rem' }}>No records found.</p>
            ) : (
                <div className="table-container">
                    <table className="table">
                        <thead>
                            <tr>
                                <th>Date</th>
                                <th>Patient</th>
                                <th>Details</th>
                                <th>Prediction</th>
                                <th>Confidence</th>
                                <th>Image</th>
                            </tr>
                        </thead>
                        <tbody>
                            {history.map(item => (
                                <tr key={item.id}>
                                    <td>{new Date(item.created_at).toLocaleDateString()}</td>
                                    <td>
                                        <div style={{ fontWeight: 600 }}>{item.patient_name}</div>
                                    </td>
                                    <td>{item.patient_gender}, {item.patient_age}</td>
                                    <td>
                                        <span className={`badge ${item.prediction === 'Fracture' ? 'badge-danger' : 'badge-success'}`}>
                                            {item.prediction}
                                        </span>
                                    </td>
                                    <td>{(item.confidence * 100).toFixed(1)}%</td>
                                    <td>
                                        <div style={{ width: '40px', height: '40px', overflow: 'hidden', borderRadius: '4px', background: '#f1f5f9' }}>
                                            {item.gradcam_image_path && (
                                                <img
                                                    src={item.gradcam_image_path}
                                                    alt="Thumb"
                                                    style={{ width: '100%', height: '100%', objectFit: 'cover' }}
                                                />
                                            )}
                                        </div>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            )}
        </div>
    );
}

export default App;
