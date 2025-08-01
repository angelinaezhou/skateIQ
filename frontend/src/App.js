import background from './background.png'
import './App.css';
import LeadingIcon from './LeadingIcon';
import Choice from './Choice'
import Skaters from './Skaters'
import List from './List';
import Existing from './Existing';
import Upload from './Upload';
import Dropdown from './Dropdown';
import SelectBox from './SelectBox';
import { Play } from 'lucide-react'
/* <img src={background} alt='background' className='background'></img> */

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
