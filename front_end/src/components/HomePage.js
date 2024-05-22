import React, { useState } from 'react';
import axios from 'axios';
import '../css/index.css'; // Import CSS file for styling

const HomePage = () => {
    const [question, setQuestion] = useState('Who is Luke Skywalker?');
    const [model, setModel] = useState('gpt2');
    const [maxNumTokens, setMaxNumTokens] = useState(50);
    const [response, setResponse] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
  
    const handleSubmit = async (e) => {
      e.preventDefault();
      setLoading(true);
      setError(null);
      try {
        const response = await axios.post('http://localhost:8000/generate', {
          question,
          model,
          max_new_tokens: maxNumTokens
        });
        setResponse(response.data.answer);
      } catch (error) {
        setError('An error occurred. Please try again.');
      } finally {
        setLoading(false);
      }
    };
  
    return (
      <div className="generate-form">
        <h2>Ask anything about  Luke Skywalker.</h2>
        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label htmlFor="question">Question:</label>
            <input type="text" id="question" value={question} onChange={(e) => setQuestion(e.target.value)} required />
          </div>
          <div className="form-group">
            <label htmlFor="model">Model:</label>
            <select id="model" value={model} onChange={(e) => setModel(e.target.value)} required>
              <option value="">Select Model</option>
              <option value="bloom">Bloom</option>
              <option value="gpt2">GPT-2</option>
              <option value="openai">openAi</option>
            </select>
          </div>
          <div className="form-group">
            <label htmlFor="maxNumTokens">Max Number of Tokens:</label>
            <input type="number" id="maxNumTokens" value={maxNumTokens} onChange={(e) => setMaxNumTokens(parseInt(e.target.value))} required />
          </div>
          <button type="submit" className="btn-submit" disabled={loading}>
            {loading ? 'Generating...' : 'Generate'}
          </button>
        </form>
        {error && <div className="error">{error}</div>}
        {response && (
          <div className="response">
            <h3>Response:</h3>
            <p>{response}</p>
          </div>
        )}
      </div>
    );
};

export default HomePage;
