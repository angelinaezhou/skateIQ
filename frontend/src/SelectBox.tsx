import "./SelectBox.css";
import { ChevronUp, ChevronDown } from "lucide-react"
import { useState } from "react";
import Dropdown from "./Dropdown";

export default function SelectBox() {
  const [isOpen, setIsOpen] = useState(false);
  const [selectedText, setSelectedText] = useState("Select your video:");
  const [selectedVideo, setSelectedVideo] = useState<{name: string, event: string, filename: string} | null>(null);
  const [showVideoModal, setShowVideoModal] = useState(false);

  const handleToggle = () => setIsOpen(!isOpen);
  
  const handleSkaterSelect = (name: string, event: string, filename: string) => {
    setSelectedText(`${name} - ${event}`);
    setSelectedVideo({ name, event, filename });
    setIsOpen(false); // Close dropdown when skater is selected
    setShowVideoModal(true); // Show video modal
  };

  const closeVideoModal = () => {
    setShowVideoModal(false);
  };

  return (
    <>
      <div className={`select-box ${isOpen ? 'open' : ''}`} onClick={handleToggle}>
        <span className="placeholder">{selectedText}</span>
        <div className="chevron">
          {isOpen ? <ChevronUp strokeWidth={3} /> : <ChevronDown strokeWidth={3} />}
        </div>
      </div>
      {isOpen && (
        <div className={`dropdown-wrapper ${isOpen ? 'open' : ''}`}>
          <Dropdown isOpen={isOpen} onSkaterSelect={handleSkaterSelect} />
        </div>
      )}
      
    </>
  );
}