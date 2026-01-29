import { useState, useRef } from 'react';
import './index.css';

interface AnalysisResult {
    prediction: string;
    confidence: string;
    probabilities: {
        Healthy: string;
        Fracture: string;
    };
    gradcam_image: string;
}

function App() {
    const [file, setFile] = useState<File | null>(null);
    const [preview, setPreview] = useState<string | null>(null);
    const [loading, setLoading] = useState(false);
    const [result, setResult] = useState<AnalysisResult | null>(null);
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

        try {
            const response = await fetch('/api/analyze', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                throw new Error('Analysis failed');
            }

            const data = await response.json();
            setResult(data);
        } catch (err) {
            setError('Failed to analyze image. Ensure backend is running.');
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="container" style={{ maxWidth: '1200px', padding: '2rem' }}>
            {/* Header */}
            <header className="animate-fade-in" style={{ textAlign: 'center', marginBottom: '3rem' }}>
                <h1 style={{ fontSize: '3.5rem', margin: 0 }} className="title-gradient">
                    Bone Fracture AI
                </h1>
                <p style={{ color: 'var(--text-dim)', fontSize: '1.2rem', marginTop: '0.5rem' }}>
                    Advanced ResNet50 Detection with Grad-CAM Visualization
                </p>
            </header>

            <main style={{ display: 'grid', gridTemplateColumns: result ? '1fr 1fr' : '1fr', gap: '2rem', transition: 'all 0.5s ease' }}>

                {/* Upload Section */}
                <section className="glass-panel animate-fade-in" style={{ padding: '2rem', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '400px' }}>
                    <input
                        type="file"
                        ref={fileInputRef}
                        onChange={handleFileChange}
                        accept="image/*"
                        style={{ display: 'none' }}
                    />

                    {preview ? (
                        <div style={{ width: '100%', height: '300px', display: 'flex', justifyContent: 'center', alignItems: 'center', overflow: 'hidden', borderRadius: '1rem', marginBottom: '1.5rem', background: 'rgba(0,0,0,0.2)' }}>
                            <img src={preview} alt="X-ray Preview" style={{ maxHeight: '100%', maxWidth: '100%', objectFit: 'contain' }} />
                        </div>
                    ) : (
                        <div
                            onClick={() => fileInputRef.current?.click()}
                            style={{
                                border: '2px dashed var(--glass-border)',
                                borderRadius: '1rem',
                                padding: '4rem',
                                cursor: 'pointer',
                                textAlign: 'center',
                                width: '80%',
                                marginBottom: '1.5rem',
                                transition: 'border-color 0.3s'
                            }}
                            onMouseEnter={(e) => e.currentTarget.style.borderColor = 'var(--primary-color)'}
                            onMouseLeave={(e) => e.currentTarget.style.borderColor = 'var(--glass-border)'}
                        >
                            <p style={{ fontSize: '1.5rem', margin: 0 }}>Drop X-ray here or click to upload</p>
                            <p style={{ color: 'var(--text-dim)' }}>Supports JPG, PNG</p>
                        </div>
                    )}

                    <div style={{ display: 'flex', gap: '1rem' }}>
                        <button
                            className="btn-primary"
                            onClick={() => fileInputRef.current?.click()}
                            style={{ background: 'transparent', border: '1px solid var(--glass-border)' }}
                        >
                            Select Image
                        </button>
                        <button
                            className="btn-primary"
                            onClick={handleAnalyze}
                            disabled={!file || loading}
                        >
                            {loading ? 'Analyzing...' : 'Run Diagnostics'}
                        </button>
                    </div>

                    {error && <p style={{ color: 'var(--danger)', marginTop: '1rem' }}>{error}</p>}
                </section>

                {/* Result Section */}
                {result && (
                    <section className="glass-panel animate-fade-in" style={{ padding: '2rem', animationDelay: '0.2s' }}>
                        <h2 style={{ marginTop: 0, marginBottom: '1.5rem', borderBottom: '1px solid var(--glass-border)', paddingBottom: '1rem' }}>
                            Diagnostic Report
                        </h2>

                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '2rem' }}>
                            <div style={{ padding: '1.5rem', background: 'rgba(0,0,0,0.2)', borderRadius: '1rem', textAlign: 'center' }}>
                                <p style={{ color: 'var(--text-dim)', fontSize: '0.9rem', marginBottom: '0.5rem' }}>PREDICTION</p>
                                <div style={{
                                    fontSize: '2rem',
                                    fontWeight: 'bold',
                                    color: result.prediction === 'Fracture' ? 'var(--danger)' : 'var(--success)'
                                }}>
                                    {result.prediction.toUpperCase()}
                                </div>
                            </div>
                            <div style={{ padding: '1.5rem', background: 'rgba(0,0,0,0.2)', borderRadius: '1rem', textAlign: 'center' }}>
                                <p style={{ color: 'var(--text-dim)', fontSize: '0.9rem', marginBottom: '0.5rem' }}>CONFIDENCE</p>
                                <div style={{ fontSize: '2rem', fontWeight: 'bold', color: 'var(--primary-color)' }}>
                                    {result.confidence}
                                </div>
                            </div>
                        </div>

                        <h3 style={{ fontSize: '1.1rem', marginBottom: '1rem' }}>Grad-CAM Visualization (AI Attention)</h3>
                        <div style={{ width: '100%', borderRadius: '1rem', overflow: 'hidden', border: '1px solid var(--glass-border)' }}>
                            <img src={result.gradcam_image} alt="Grad-CAM" style={{ width: '100%', display: 'block' }} />
                        </div>

                        <div style={{ marginTop: '1.5rem', fontSize: '0.9rem', color: 'var(--text-dim)' }}>
                            Probabilities: Healthy ({result.probabilities.Healthy}) | Fracture ({result.probabilities.Fracture})
                        </div>
                    </section>
                )}

            </main>
        </div>
    );
}

export default App;
