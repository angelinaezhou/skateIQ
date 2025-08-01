import { useRef, useState } from "react"
import "./Own.css"
import LeadingIcon from "./LeadingIcon"
import { Upload, CheckCircle, AlertCircle, Clock, Play, X, RotateCcw } from "lucide-react"

type OwnProps = {
  onUploadSuccess?: (result: any) => void;
  onUploadError?: (error: string) => void;
  onClassificationResult?: (result: any) => void;
};

type ComponentState = 'upload' | 'processing' | 'video-ready' | 'playing-video' | 'results';

type UploadResult = {
  video_id: string;
  original_filename: string;
  pose_filename: string;
  video_info?: any;
  jump_type?: string;
  confidence?: number;
  all_probabilities?: { [key: string]: number };
  classification?: string;
};

export default function Own({ onUploadSuccess, onUploadError, onClassificationResult }: OwnProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [state, setState] = useState<ComponentState>('upload');
  const [uploadResult, setUploadResult] = useState<UploadResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Create video URL for uploaded file (you'll need to serve uploaded videos)
  const getVideoUrl = (videoId: string) => {
    return `http://localhost:8000/uploads/${videoId}.mp4`; // Adjust based on your backend
  };

  const handleClick = () => {
    if (state === 'upload') {
      // Trigger file upload
      fileInputRef.current?.click();
    } else if (state === 'video-ready' || state === 'results') {
      // Play the video
      setState('playing-video');
    }
  };

  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    // Validate file type
    const allowedTypes = ['video/mp4', 'video/quicktime', 'video/mov', 'video/x-msvideo'];
    if (!allowedTypes.includes(file.type)) {
      setError('Please select a valid video file (.mp4, .mov, .avi)');
      return;
    }

    // Validate file size
    const maxSize = 500 * 1024 * 1024; // 500MB
    if (file.size > maxSize) {
      setError('File size must be less than 500MB');
      return;
    }

    await uploadAndProcessVideo(file);
  };

  const uploadAndProcessVideo = async (file: File) => {
    setState('processing');
    setError(null);

    try {
      const formData = new FormData();
      formData.append('video', file);
      formData.append('skater', 'User');
      formData.append('event', 'Upload');

      console.log('🚀 Starting video upload and processing...');
      console.log('🔗 Requesting:', 'http://localhost:8000/api/upload-and-classify');

      const response = await fetch('http://localhost:8000/api/upload-and-classify', {
        method: 'POST',
        body: formData,
      });

      console.log('📡 Response status:', response.status);
      console.log('📡 Response headers:', response.headers.get('content-type'));

      if (!response.ok) {
        // Try to get error message from response
        let errorMessage = `HTTP ${response.status}: ${response.statusText}`;
        
        try {
          const contentType = response.headers.get('content-type');
          if (contentType && contentType.includes('application/json')) {
            const errorData = await response.json();
            errorMessage = errorData.error || errorMessage;
          } else {
            const textError = await response.text();
            console.log('📄 Non-JSON error response:', textError);
            errorMessage = textError || errorMessage;
          }
        } catch (parseError) {
          console.log('⚠️ Could not parse error response:', parseError);
        }
        
        throw new Error(errorMessage);
      }

      // Check if response is JSON
      const contentType = response.headers.get('content-type');
      if (!contentType || !contentType.includes('application/json')) {
        const textResponse = await response.text();
        console.log('📄 Non-JSON response:', textResponse);
        throw new Error('Server returned non-JSON response. Check if the backend is running correctly.');
      }

      const result = await response.json();
      console.log('✅ Upload result:', result);
      
      setUploadResult(result);
      setState('video-ready');
      
      onUploadSuccess?.(result);

    } catch (error) {
      console.error('❌ Upload error:', error);
      let errorMessage = 'Upload failed';
      
      if (error instanceof TypeError && error.message.includes('fetch')) {
        errorMessage = 'Cannot connect to server. Is the backend running on port 8000?';
      } else if (error instanceof Error) {
        errorMessage = error.message;
      }
      
      setError(errorMessage);
      setState('upload');
      onUploadError?.(errorMessage);
    } finally {
      // Reset file input
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const handleVideoEnd = async () => {
    if (!uploadResult) return;
    
    setState('processing');
    
    try {
      const response = await fetch('http://localhost:8000/api/classify-jump', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          filename: uploadResult.pose_filename,
          skater: 'User',
          event: 'Upload'
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || `Classification failed: ${response.statusText}`);
      }

      const classificationResult = await response.json();
      console.log('🎯 Classification result:', classificationResult);
      
      // Merge classification results with upload result
      const finalResult = {
        ...uploadResult,
        ...classificationResult
      };
      
      setUploadResult(finalResult);
      setState('results');
      onClassificationResult?.(finalResult);

    } catch (error) {
      console.error('❌ Classification error:', error);
      setError(error instanceof Error ? error.message : 'Classification failed');
      setState('video-ready');
    }
  };

  const closeVideo = () => {
    setState(uploadResult?.jump_type ? 'results' : 'video-ready');
  };

  const resetComponent = () => {
    setState('upload');
    setUploadResult(null);
    setError(null);
  };

  const getIcon = () => {
    switch (state) {
      case 'upload': return Upload;
      case 'processing': return Clock;
      case 'video-ready': return Play;
      case 'playing-video': return Play;
      case 'results': return CheckCircle;
      default: return Upload;
    }
  };

  const getIconColor = () => {
    switch (state) {
      case 'processing': return '#f59e0b'; // amber
      case 'results': return '#10b981'; // green
      case 'video-ready': return '#3b82f6'; // blue
      default: return undefined;
    }
  };

  const getTitle = () => {
    switch (state) {
      case 'upload': return 'Upload from computer';
      case 'processing': return 'Processing video...';
      case 'video-ready': return `Ready: ${uploadResult?.original_filename}`;
      case 'playing-video': return 'Playing video...';
      case 'results': return `Detected: ${uploadResult?.jump_type}`;
      default: return 'Upload from computer';
    }
  };

  const getSubtitle = () => {
    switch (state) {
      case 'upload': return '.mp4, .mov, .avi files allowed';
      case 'processing': return 'Analyzing with AI...';
      case 'video-ready': return 'Click to play and analyze';
      case 'playing-video': return 'Video will auto-classify when finished';
      case 'results': return `Confidence: ${uploadResult?.confidence ? (uploadResult.confidence * 100).toFixed(1) : 0}%`;
      default: return '.mp4, .mov, .avi files allowed';
    }
  };

  const isClickable = state === 'upload' || state === 'video-ready' || state === 'results';

  return (
    <>
      <div 
        className={`own ${state} ${error ? 'error' : ''}`}
        onClick={isClickable ? handleClick : undefined}
        style={{ cursor: isClickable ? 'pointer' : 'default' }}
      >
        <LeadingIcon 
          icon={getIcon()} 
          size={20} 
          strokeWidth={4}
          color={getIconColor()}
        />
        <div className="text">
          <div className='title'>{getTitle()}</div>
          <div className="subtitle">{getSubtitle()}</div>
        </div>
        
        {(state === 'results' || error) && (
          <button 
            className="reset-button" 
            onClick={(e) => {
              e.stopPropagation();
              resetComponent();
            }}
          >
            <RotateCcw size={16} />
          </button>
        )}
      </div>
      
      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept="video/mp4,video/quicktime,video/mov,video/x-msvideo,.mp4,.mov,.avi"
        onChange={handleFileChange}
        style={{ display: 'none' }}
        disabled={state === 'processing'}
      />

      {/* Video Modal */}
      {state === 'playing-video' && uploadResult && (
        <div className="video-modal">
          <div className="video-container">
            <button className='close-button' onClick={closeVideo}>
              <X />
            </button>

            <video 
              className="video"
              src={getVideoUrl(uploadResult.video_id)}
              controls
              autoPlay
              onEnded={handleVideoEnd}
              style={{width: '100%', maxWidth:'500px'}}
            >
              Your browser does not support the video tag.
            </video>

            <div className="video-info">
              <h4>{uploadResult.original_filename}</h4>
              <p>Video will be analyzed when playback ends</p>
            </div>
          </div>
        </div>
      )}

      {/* Error Display */}
      {error && (
        <div className="error-message">
          <AlertCircle size={16} color="#ef4444" />
          <span>{error}</span>
        </div>
      )}
    </>
  );
}