import './Choice.css';

type ChoiceProps = { 
    title?: string, 
    subtitle?: string
}

export default function Choice({title="Title", subtitle="Subtitle"} : ChoiceProps) {
    return (
        <div className='choice'>
            <div className='title'>{title}</div>
            <div className='subtitle'>{subtitle}</div>
        </div>
    )
}