import background from './background.png'
import './App.css';
import SelectBox from './SelectBox';

function App() {
  return (
    <div className="App">
      <img src={background} className='background'/>
      <div className='test'>
        <SelectBox />
      </div>
    </div>
  );
}

export default App;
