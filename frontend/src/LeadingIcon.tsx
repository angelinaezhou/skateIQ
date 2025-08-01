import { type LucideIcon } from "lucide-react";
import "./LeadingIcon.css";

type LeadingIconProps = {
    icon: LucideIcon; 
    size?: number;
    color?: string;
    strokeWidth?: number;
};

export default function LeadingIcon({
    icon: Icon,
    size = 23,
    color = "#5296B8",
    strokeWidth = 3
}: LeadingIconProps) {
    return (
        <div className='square'>
            <div className="leading-icon">
                <Icon size={size} color={color} strokeWidth={strokeWidth}/>
            </div>
        </div>
    );
}
