/* each of the skater's individual selection */
import "./Skaters.css"
import LeadingIcon from "./LeadingIcon";
import { Play, X, Video, SquarePlay, Clapperboard } from "lucide-react"
import { useState } from "react"

type skatersProps = {
  name: string;
  event: string;
  filename: string;
  onSelect?: (name: string, event: string, filename: string) => void;
};

type ClassificationResult = {
  jump_type: string;
  confidence: number;
  all_probabilities?: { [key: string]: number};
  top_predictions?: Array<{
    rank: number;
    jump_type: string; 
    probability: number;
  }>;
} | null;

export default function Skaters({ name, event, filename, onSelect }: skatersProps) {
  const [showVideo, setShowVideo] = useState(false);
  const [classificationResult, setClassificationResult] = useState<ClassificationResult>(null);
  const [isProcessing, setIsProcessing] = useState(false);

  const getVideoPath = (name: string, event: string) => {
    const videoMap: { [key: string]: string } = {
      "Yuna Kim-2010 Olympics Free Skate": "/programs/yuna_lutz.mp4",
      "Alina Zagitova-2018 Olympics Short Program": "/programs/alina_loop.mp4",
      "Yuzuru Hanyu-2018 Olympics Short Program": "/programs/yuzuru_axel.mp4",
      "Yuzuru Hanyu-2018 Olympics Free Skate": "/programs/yuzuru_sal.mp4",
      "Nathan Chen-2022 Olympics Free Skate": "/programs/nathan_flip.mp4",
      "Yuna Kim-2014 Olympics Short Program": "/programs/yuna_toe.mp4",
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
      setClassificationResult(result);
      
    } catch (error) {
      console.error('❌ Classification failed:', error);
      setClassificationResult({
        jump_type: 'Classification failed - check backend connection',
        confidence: 0,
      });
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
        <LeadingIcon icon={Clapperboard} size={20} strokeWidth={2.9} />
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
              onEnded={handleVideoEnd}
            >
              Your browser does not support the video tag.
            </video>
            {classificationResult && (
              <div className="classification-result">
                {classificationResult.top_predictions && (
                  <div className="top-predictions">
                    <h4>top predictions:</h4>
                    {classificationResult.top_predictions.slice(0, 3).map((pred, index) => (
                      <div key={index} className="prediction-item">
                        <div className="prediction-header">
                          <span className="prediction-rank">#{pred.rank}</span>
                          <span className="prediction-name">{pred.jump_type.toLowerCase()}</span>
                          <span className="prediction-percentage">{(pred.probability * 100).toFixed(1)}%</span>
                        </div>
                        <div className="confidence-bar">
                          <div 
                            className="confidence-fill" 
                            style={{
                              width: `${pred.probability * 100}%`,
                              backgroundColor: index === 0 ? '#698DE9' : index === 1 ? '#AEC1F0' : '#D9E1F6'
                            }}
                          ></div>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </>
  );
}