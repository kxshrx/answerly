document.getElementById("askBtn").addEventListener("click", async () => {
  const questionInput = document.getElementById("question");
  const answerDiv = document.getElementById("answer");
  const question = questionInput.value.trim();

  answerDiv.textContent = "";  // clear previous answer

  if (!question) {
    answerDiv.textContent = "Please enter a question.";
    return;
  }

  try {
    // Get current active tab URL
    let [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    let currentUrl = tab.url;

    // Show loading indicator
    answerDiv.textContent = "Loading...";

    // Call your backend
    const response = await fetch("http://localhost:8000/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, url: currentUrl })
    });

    if (!response.ok) {
      throw new Error(`Server error: ${response.statusText}`);
    }

    const data = await response.json();

    answerDiv.textContent = `Answer:\n${data.answer}\n\nSource:\n${data.source}`;
  } catch (error) {
    answerDiv.textContent = `Error: ${error.message}`;
  }
});
