import HoldemGame from "./HoldemGame";
import "./holdem.css";

export default function App() {
  return (
    <>
      <header className="lab-nav">
        <a className="lab-brand" href="/">
          <span>♠</span> HOLDEMBOT
        </a>
        <span className="lab-local">LOCAL PLAY</span>
      </header>
      <HoldemGame />
    </>
  );
}
