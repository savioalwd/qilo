import logo from './logo.svg';
import './App.css';


import HomePage from './components/HomePage';
import Footer from './components/footer';
function App() {
  return (
    <div className="app">
      <div className="center-container">
        <HomePage />
      </div>
      <Footer />
    </div>
  );
}

export default App;
