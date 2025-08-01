import "./Dropdown.css"
import Existing from "./Existing"
import Upload from "./Upload";

type DropdownProps = {
  isOpen: boolean;
  onSkaterSelect?: (name: string, event: string, filename: string) => void; // Updated to include filename
};

export default function Dropdown({ isOpen, onSkaterSelect }: DropdownProps) {
  return (
    <div className={`dropdown ${isOpen ? 'open' : ''}`}>
      <div className="upload-section"><Upload/></div>
      <div className="existing-section"><Existing onSkaterSelect={onSkaterSelect} /></div>
    </div>
  );
}