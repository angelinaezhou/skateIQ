/* each of the skater's individual selection */
import "./Skaters.css"
import LeadingIcon from "./LeadingIcon";
import { Play, X } from "lucide-react"
import { useState } from "react"

type skatersProps = {
  name: string;
  event: string;
  filename: string;
  onSelect?: (name: string, event: string, filename: string) => void;
};

export default function Skaters({ name, event, filename, onSelect }: skatersProps) {
  const [showVideo, setShowVideo] = useState(false);
  const [classificationResult, setClassificationResult] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);

  const getVideoPath = (name: string, event: string) => {
    const videoMap: { [key: string]: string } = {
      "Yuna Kim-2010 Olympics Free Skate": "/programs/yuna_2010.mp4",
      "Alina Zagitova-2018 Olympics Short Program": "/programs/alina_2018.mp4",
      "Yuzuru Hanyu-2018 Olympics Short Program": "/programs/yuzuru_2018.mp4",
      "Yulia Lipnitskaya-2014 Olympics Free Skate": "/programs/yulia_2014.mp4",
      "Nathan Chen-2022 Olympics Short Program": "/programs/nathan_2022.mp4",
      "Yuna Kim-2014 Olympics Short Program": "/programs/yuna_2014.mp4",
    };
    return videoMap[`${name}-${event}`] || "";
  }

  const handleClick = () => {
    console.log('🎯 Before click - showVideo:', showVideo);
    const videoPath = getVideoPath(name, event);
    console.log('Video path:', videoPath);
    console.log('Full key:', `${name}-${event}`);
    setShowVideo(true);
    console.log('🎯 After click - should be true', showVideo);
    setClassificationResult(null);
  };

  const handleVideoEnd = async () => {
    setIsProcessing(true);
    try {
      console.log('🚀 Making API call to classify jump...');
      
      const response = await fetch('http://localhost:8000/api/classify-jump', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          filename: filename,
          skater: name,
          event: event
        }),
      });

      console.log('📡 Response status:', response.status);
      
      if (!response.ok) {
        const errorText = await response.text();
        console.log('❌ Error response:', errorText);
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      console.log('✅ Classification result:', result);
      setClassificationResult(result.classification);
      
    } catch (error) {
      console.error('❌ Classification failed:', error);
      setClassificationResult('Classification failed - check backend connection');
    } finally {
      setIsProcessing(false);
    }
  };

  const closeVideo = () => {
    setShowVideo(false);
    setClassificationResult(null);
    setIsProcessing(false);
    if (onSelect) {
      onSelect(name, event, filename);
    }
  }

  return (
    <>
      <div className="skater" onClick={handleClick} style={{cursor: 'pointer'}}>
        <LeadingIcon icon={Play} size={20} strokeWidth={4} />
        <div className="description">
          <div className="title">{name}</div>
          <div className="event">{event}</div>
        </div>
      </div>
      {showVideo && (
        <div className="video-modal">
          <div className="video-container">
            <button className='close-button' onClick={closeVideo}><X/></button>
            <video className="video"
              src={getVideoPath(name, event)}
              controls
              autoPlay
              onEnded={handleVideoEnd}
              style={{width: '100%', maxWidth:'500px'}}
            >
              Your browser does not support the video tag.
            </video>
            {isProcessing && (
              <div className="processing">
                <p>Analyzing jump...</p>
                <div className="spinner"></div>
              </div>
            )}
            {classificationResult && (
              <div className='classification-result'>
                <h3>Jump Detected:</h3>
                <p>{classificationResult}</p>
              </div>
            )}
          </div>
        </div>
      )}
    </>
  );
}