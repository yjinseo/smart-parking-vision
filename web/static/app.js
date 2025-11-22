window.addEventListener("DOMContentLoaded", () => {
    const box = document.getElementById("predict-data");
    if (!box) return;

    // HTML 속성에서 JSON 읽기
    const predicted = JSON.parse(box.dataset.predicted);
    const history = JSON.parse(box.dataset.history);

    const ctx = document.getElementById("predictChart").getContext("2d");

    const labels = ["현재", "+10분", "+20분", "+30분", "+40분", "+50분", "+60분"];

    new Chart(ctx, {
        type: "line",
        data: {
            labels: labels,
            datasets: [
                {
                    label: "예상 점유 대수",
                    borderColor: "#4A90E2",
                    backgroundColor: "rgba(74,144,226,0.2)",
                    data: predicted,
                    tension: 0.3,
                    fill: true
                },
                {
                    label: "과거 평균 점유",
                    borderColor: "#F5A623",
                    backgroundColor: "rgba(245,166,35,0.2)",
                    data: history,
                    tension: 0.3,
                    fill: true
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false
        }
    });
});

