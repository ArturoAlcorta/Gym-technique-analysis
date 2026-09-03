const STAGE_LABELS = {
  preprocess: "Resampling video",
  pose: "Estimating pose",
  normalize: "Normalizing keypoints",
  reps: "Counting reps",
  technique: "Measuring technique",
  compare: "Comparing with references",
  done: "Done",
  error: "Failed",
};

function watchAnalysis(analysisId, onEvent) {
  const source = new EventSource(`/analyses/${analysisId}/events`);
  source.onmessage = (event) => {
    const payload = JSON.parse(event.data);
    onEvent(payload);
    if (payload.status === "done" || payload.status === "error") {
      source.close();
    }
  };
  source.onerror = () => source.close();
}
