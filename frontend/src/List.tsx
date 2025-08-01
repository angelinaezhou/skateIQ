import Skaters from "./Skaters";
import "./List.css"

type ListProps = {
  onSkaterSelect?: (name: string, event: string, filename: string) => void; // Updated
};

export default function List({ onSkaterSelect }: ListProps) {
  return (
    <div className="skaters-grid">
      <Skaters name="Yuna Kim" event="2010 Olympics Free Skate" filename="lutz_04.npy" onSelect={onSkaterSelect}/>
      <Skaters name="Alina Zagitova" event="2018 Olympics Short Program" filename="lutz_04.npy" onSelect={onSkaterSelect}/>
      <Skaters name="Yuzuru Hanyu" event="2018 Olympics Short Program" filename="lutz_04.npy" onSelect={onSkaterSelect}/>
      <Skaters name="Yulia Lipnitskaya" event="2014 Olympics Free Skate" filename="lutz_04.npy" onSelect={onSkaterSelect}/>
      <Skaters name="Nathan Chen" event="2022 Olympics Short Program" filename="lutz_04.npy" onSelect={onSkaterSelect}/>
      <Skaters name="Yuna Kim" event="2014 Olympics Short Program" filename="lutz_04.npy" onSelect={onSkaterSelect}/>
    </div>
  );
}