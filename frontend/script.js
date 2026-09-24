const API_URL = "http://127.0.0.1:5000/api";

let predictionHistory = [];

const predictButton =document.querySelector(".predict-btn");
const predictionForm = document.getElementById("predictionForm");
const loadingCard = document.getElementById("loadingCard");
const resultCard = document.getElementById("resultCard");

const predictionResult = document.getElementById("predictionResult");
const predictionStatus =document.getElementById("predictionStatus");
const predictionMessage = document.getElementById("predictionMessage");

const failureProbability = document.getElementById("failureProbability");
const normalProbability = document.getElementById("normalProbability");

const failureProgress = document.getElementById("failureProgress");
const normalProgress = document.getElementById("normalProgress");

const resetButton = document.getElementById("resetButton");

const totalPredictions = document.getElementById("totalPredictions");
const normalPredictions = document.getElementById("normalPredictions");
const failurePredictions = document.getElementById("failurePredictions");
const apiStatus = document.getElementById("apiStatus");

const historyContainer = document.getElementById("historyContainer");
const featureImportanceContainer =
    document.getElementById("featureImportance");


/* ================================
   API Health Check
================================ */

async function checkAPIStatus() {

    try {

        const response = await fetch(`${API_URL}/health`);

        if (!response.ok) {
            throw new Error("API unavailable");
        }

        apiStatus.textContent = "Online";

    } catch (error) {

        apiStatus.textContent = "Offline";

    }
}


/* ================================
   Feature Importance
================================ */

async function loadFeatureImportance() {

    try {

        const response =
            await fetch(`${API_URL}/feature-importance`);

        const data = await response.json();

        if (data.status !== "success") {
            throw new Error("Unable to load feature importance");
        }

        renderFeatureImportance(data.features);

    } catch (error) {

        featureImportanceContainer.innerHTML = `
            <p class="empty-history">
                Unable to load feature importance.
            </p>
        `;

        console.error("Feature importance error:", error);
    }
}


function renderFeatureImportance(features) {

    featureImportanceContainer.innerHTML = "";

    const maxImportance = Math.max(
        ...features.map(function (item) {
            return item.Importance;
        })
    );

    features.forEach(function (item) {

        const percentage =
            (item.Importance / maxImportance) * 100;

        const featureItem = document.createElement("div");

        featureItem.className = "feature-item";

        featureItem.innerHTML = `
            <div class="feature-info">
                <span class="feature-name">
                    ${item.Feature}
                </span>

                <span class="feature-value">
                    ${(item.Importance * 100).toFixed(2)}%
                </span>
            </div>

            <div class="feature-bar">
                <div
                    class="feature-fill"
                    style="width: ${percentage}%;">
                </div>
            </div>
        `;

        featureImportanceContainer.appendChild(featureItem);

    });
}


/* ================================
   Prediction
================================ */

predictionForm.addEventListener(
    "submit",
    async function (event) {

        event.preventDefault();
        const predictButton = document.querySelector(".predict-btn");
        predictButton.classList.add("loading");
        predictButton.disabled = true;
        predictButton.textContent = "Analyzing";

        const machineData = {

            "Type":
                document.getElementById("type").value,

            "Air temperature [K]":
                parseFloat(
                    document.getElementById("airTemperature").value
                ),

            "Process temperature [K]":
                parseFloat(
                    document.getElementById("processTemperature").value
                ),

            "Rotational speed [rpm]":
                parseFloat(
                    document.getElementById("rotationalSpeed").value
                ),

            "Torque [Nm]":
                parseFloat(
                    document.getElementById("torque").value
                ),

            "Tool wear [min]":
                parseFloat(
                    document.getElementById("toolWear").value
                )
        };


        loadingCard.style.display = "block";
        resultCard.style.display = "none";


        try {

            const response = await fetch(
                `${API_URL}/predict`,
                {
                    method: "POST",

                    headers: {
                        "Content-Type": "application/json"
                    },

                    body: JSON.stringify(machineData)
                }
            );


            const data = await response.json();


            if (!response.ok) {
                throw new Error(data.message || "Prediction failed");
            }


            displayPrediction(data);

            addPredictionToHistory(
                machineData,
                data
            );


        } catch (error) {

            alert(error.message);

        } finally {

            loadingCard.style.display = "none";
            predictButton.classList.remove("loading");
            predictButton.disabled = false;
            predictButton.textContent = "Predict Machine Failure";

        }

    }
);


/* ================================
   Display Prediction
================================ */

function displayPrediction(data) {

    resultCard.style.display = "block";


    predictionResult.textContent =
        data.result;

    if (data.prediction === 1) {

        predictionStatus.textContent =
        "⚠ Failure Risk Detected";

        predictionStatus.style.background =
        "#fee2e2";

        predictionStatus.style.color =
        "#b91c1c";
    } else {    

        predictionStatus.textContent =
        "✓ Normal Operation";

        predictionStatus.style.background =
        "#dcfce7";

        predictionStatus.style.color =
        "#15803d";

    }

    if (data.prediction === 1) {

        predictionMessage.textContent =
            "The model predicts that this machine is likely to experience a failure.";

    } else {

        predictionMessage.textContent =
            "The model predicts that the machine is likely to operate normally.";

    }


    failureProbability.textContent =
        `${data.failure_probability}%`;

    normalProbability.textContent =
        `${data.normal_probability}%`;


    failureProgress.style.width =
        `${data.failure_probability}%`;

    normalProgress.style.width =
        `${data.normal_probability}%`;


    if (data.prediction === 1) {

        resultCard.style.borderLeftColor =
            "#dc2626";

        failureProgress.style.background =
            "#dc2626";

        normalProgress.style.background =
            "#22c55e";

    } else {

        resultCard.style.borderLeftColor =
            "#16a34a";

        failureProgress.style.background =
            "#dc2626";

        normalProgress.style.background =
            "#22c55e";

    }

}


/* ================================
   Prediction History
================================ */

function addPredictionToHistory(
    machineData,
    predictionData
) {

    const historyItem = {

        time:
            new Date().toLocaleTimeString(),

        type:
            machineData["Type"],

        failureProbability:
            predictionData.failure_probability,

        prediction:
            predictionData.prediction

    };


    predictionHistory.unshift(historyItem);


    if (predictionHistory.length > 10) {
        predictionHistory.pop();
    }


    updateDashboardStats();

    renderPredictionHistory();

}


/* ================================
   Dashboard Statistics
================================ */

function updateDashboardStats() {

    const total =
        predictionHistory.length;


    const failures =
        predictionHistory.filter(
            function (item) {
                return item.prediction === 1;
            }
        ).length;


    const normal =
        total - failures;


    totalPredictions.textContent =
        total;

    normalPredictions.textContent =
        normal;

    failurePredictions.textContent =
        failures;

}


/* ================================
   Render History
================================ */

function renderPredictionHistory() {

    if (predictionHistory.length === 0) {

        historyContainer.innerHTML = `
            <p class="empty-history">
                No predictions yet.
            </p>
        `;

        return;
    }


    historyContainer.innerHTML = "";


    predictionHistory.forEach(
        function (item) {

            const historyElement =
                document.createElement("div");


            historyElement.className =
                "history-item";


            const status =
                item.prediction === 1
                    ? "Failure Predicted"
                    : "Normal";


            historyElement.innerHTML = `

                <span>
                    ${item.time}
                </span>

                <span>
                    Type ${item.type}
                </span>

                <span>
                    ${item.failureProbability}%
                    failure probability
                </span>

                <strong>
                    ${status}
                </strong>

            `;


            historyContainer.appendChild(
                historyElement
            );

        }
    );

}


/* ================================
   Reset
================================ */

resetButton.addEventListener(
    "click",
    function () {

        resultCard.style.display =
            "none";

        predictionForm.reset();

    }
);


/* ================================
   Initial Load
================================ */

checkAPIStatus();

loadFeatureImportance();

renderPredictionHistory();

updateDashboardStats();