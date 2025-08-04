import Skaters from "./Skaters";
import "./List.css"

type ListProps = {
  onSkaterSelect?: (name: string, event: string, filename: string) => void; 
};

export default function List({ onSkaterSelect }: ListProps) {
  return (
    <div className="skaters-grid">
      <Skaters name="Yuna Kim" event="2010 Olympics Free Skate" filename="lutz_76.npy" onSelect={onSkaterSelect}/> 
      <Skaters name="Yuzuru Hanyu" event="2018 Olympics Short Program" filename="axel_06.npy" onSelect={onSkaterSelect}/> 
      <Skaters name="Yuzuru Hanyu" event="2018 Olympics Free Skate" filename="sal_05.npy" onSelect={onSkaterSelect}/>
      <Skaters name="Alina Zagitova" event="2018 Olympics Short Program" filename="loop_09.npy" onSelect={onSkaterSelect}/> 
      <Skaters name="Nathan Chen" event="2022 Olympics Free Skate" filename="flip_54.npy" onSelect={onSkaterSelect}/>
      <Skaters name="Yuna Kim" event="2014 Olympics Short Program" filename="toe_19.npy" onSelect={onSkaterSelect}/> 
    </div>
  );
}