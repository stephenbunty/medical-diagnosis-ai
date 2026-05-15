import {
  ArcElement,
  BarElement,
  CategoryScale,
  Chart as ChartJS,
  Legend,
  LinearScale,
  Tooltip,
} from "chart.js";
import { Bar, Doughnut } from "react-chartjs-2";

ChartJS.register(CategoryScale, LinearScale, BarElement, Tooltip, Legend, ArcElement);

export function SeverityBar({ value }: { value: number }) {
  const data = {
    labels: ["Severity"],
    datasets: [
      {
        label: "Score",
        data: [Math.min(100, Math.round(value * 100))],
        backgroundColor: "rgba(37, 99, 235, 0.6)",
      },
    ],
  };
  return (
    <div className="h-48">
      <Bar
        data={data}
        options={{
          responsive: true,
          maintainAspectRatio: false,
          scales: { y: { beginAtZero: true, max: 100 } },
        }}
      />
    </div>
  );
}

export function ModalityDoughnut({ chest, brain }: { chest: number; brain: number }) {
  const data = {
    labels: ["Chest X-ray", "Brain MRI"],
    datasets: [
      {
        data: [chest, brain],
        backgroundColor: ["#3b82f6", "#10b981"],
      },
    ],
  };
  return (
    <div className="mx-auto h-52 w-52">
      <Doughnut data={data} options={{ plugins: { legend: { position: "bottom" } } }} />
    </div>
  );
}
