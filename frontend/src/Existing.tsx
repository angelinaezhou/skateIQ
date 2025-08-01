import List from "./List"; // Adjust import path as needed

type ExistingProps = {
  onSkaterSelect?: (name: string, event: string, filename: string) => void; // Updated
};

export default function Existing({ onSkaterSelect }: ExistingProps) {
  return (
    <div>
      <div className="header">Choose existing video</div>  
      <List onSkaterSelect={onSkaterSelect} />
    </div>
  );
}