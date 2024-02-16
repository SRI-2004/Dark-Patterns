const endpoint = "http:/127.0.0.1:5000/";
const descriptions = {
  Sneaking:
    "Coerces users to act in ways that they would not normally act by obscuring information.",
  Urgency: "Places deadlines on things to make them appear more desirable",
  Misdirection:
    "Aims to deceptively incline a user towards one choice over the other.",
  "Social Proof":
    "Gives the perception that a given action or product has been approved by other people.",
  Scarcity:
    "Tries to increase the value of something by making it appear to be limited in availability.",
  Obstruction:
    "Tries to make an action more difficult so that a user is less likely to do that action.",
  "Forced Action":
    "Forces a user to complete extra, unrelated tasks to do something that should be simple.",
};

function scrape() {
  console.log("scrapping")
  // aggregate all DOM elements on the page
  let elements = segments(document.body);
  let filtered_elements = [];
  let filtered_elements_data = [];
  let patternCounts = {};
  let siteUrl = window.location.href;
  let text;

  for (let i = 0; i < elements.length; i++) {
    if (elements[i].innerText) {
      let text = elements[i].innerText.trim().replace(/\t/g, " ");
      if (text.length > 0) {
        filtered_elements.push(text);
        filtered_elements_data.push(elements[i]);
      }
    }
  }

  console.log("fetching", filtered_elements.length, filtered_elements);

  // post to the web server
  fetch(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ tokens: filtered_elements, site_url: siteUrl }),
  })
    .then((resp) => resp.json())
    .then((data) => {
      data = data.replace(/'/g, '"');
      json = JSON.parse(data);
      console.log(json);

      let dp_counts = {}; // To store counts for each pattern
      let element_index = 0;
      let text;
     for (let i = 0; i < filtered_elements_data.length; i++) {
    let element = filtered_elements_data[i];
    if (element.innerText) {
        let text = element.innerText.trim().replace(/\t/g, " ");
        if (text.length > 0 && json.result[i] !== "Not Dark") {
            dp_counts[json.result[i]] = (dp_counts[json.result[i]] || 0) + 1;
            console.log(element, json.result[i])
            highlight(element, json.result[i]);
        }
    }
}


      // store counts of dark patterns
      let g = document.createElement("div");
      g.id = "insite_count";
      g.value = dp_counts;
      g.style.opacity = 0;
      g.style.position = "fixed";
      document.body.appendChild(g);
      sendDarkPatterns(dp_counts);
    })
    .catch((error) => {
      console.error(error);
      console.error(error.stack);
    });

  let mutationBatch = []; // Array to store mutations in a batch
  const batchSize = 100; // Batch size for processing mutations

  const mutationCallback = function (mutationsList, observer) {
    for (const mutation of mutationsList) {
      if (mutation.type === "childList") {
        // Process added nodes
        mutation.addedNodes.forEach((node) => {
          if (node.nodeType === Node.ELEMENT_NODE) {
            mutationBatch.push(node); // Push node to the batch
          }
        });
      }
    }

    // If the batch size is reached, process the batch and clear it
    if (mutationBatch.length >= batchSize) {
      processBatch();
    }
  };

  // Create a Mutation Observer instance
  const observer = new MutationObserver(mutationCallback);

  // Start observing the body for changes
  observer.observe(document.body, { childList: true, subtree: true });

  // Function to process the mutation batch
  function processBatch() {
    mutationBatch.forEach((node) => {
      segmentAndFetch(node); // Process each node in the batch
    });
    mutationBatch = []; // Clear the batch after processing
  }

  // Function to segment incoming changes and fetch
  function segmentAndFetch(node) {
    // Check if the node is an element with inner text
    if (node.innerText && node.innerText.trim().length > 0) {
      const filtered_elements = [node.innerText.trim().replace(/\t/g, " ")];
      const siteUrl = window.location.href;

      // Post to the web server
      fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tokens: filtered_elements, site_url: siteUrl }),
      })
        .then((resp) => resp.json())
        .then((data) => {
          // Process response data
          data = data.replace(/'/g, '"');
          const json = JSON.parse(data);
          const dp_counts = {}; // To store counts for each pattern
          const result = json.result;

          for (let i = 0; i < result.length; i++) {
            if (result[i] !== "Not Dark") {
              dp_counts[result[i]] = (dp_counts[result[i]] || 0) + 1;
              highlight(node, result[i]);
            }
          }

          // Store counts of dark patterns
          let g = document.createElement("div");
          g.id = "insite_count";
          g.value = dp_counts;
          g.style.opacity = 0;
          g.style.position = "fixed";
          document.body.appendChild(g);
          sendDarkPatterns(dp_counts);
        })
        .catch((error) => {
          console.error(error);
        });
    }
  }
}

function highlight(element, type) {
  element.classList.add("insite-highlight");

  if (type && type.trim() !== "") {
    // Remove spaces from the type variable
    const tempType = type.replace(/\s+/g, "");
    // Add the modified type to the class list of the element
    element.classList.add(tempType);
  }

  let body = document.createElement("span");
  body.classList.add("insite-highlight-body");

  // Define background colors for each dark pattern count
  const backgroundColors = {
    Sneaking: "#3498db",
    Urgency: "#e74c3c",
    Misdirection: "#2ecc71",
    SocialProof: "#000048",
    Scarcity: "#9b59b6",
    Obstruction: "#1abc9c",
    ForcedAction: "#FF5733",
  };

  // Set background color based on the dark pattern type

  element.style.background = backgroundColors[type];

  // Add event listeners for mouseover and mouseout events
}

function sendDarkPatterns(counts) {
  chrome.runtime.sendMessage({
    message: "update_dark_pattern_counts",
    counts: counts,
  });
}

const send_feedback = (feedbackValue, selectedPattern) => {
  const feedbackData = {
    feedback: feedbackValue,
    pattern: selectedPattern,
    site_url: window.location.href,
  };

  fetch(endpoint + "feedback", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(feedbackData),
  })
    .then((response) => response.json())
    .then((data) => {
      console.log("Success:", data);
      // You can add any further handling here
    })
    .catch((error) => {
      console.error("Error:", error);
      // You can add error handling here
    });
};

chrome.runtime.onMessage.addListener(function (request, sender, sendResponse) {
  console.log(request);
  if (request.message === "analyze_site") {
    scrape();
  } else if (request.message === "popup_open") {
    let element = document.getElementById("insite_count");
    if (element) {
      sendDarkPatterns(element.value);
    }
  } else if (request.message === "feedback") {
    send_feedback(request.feedbackValue, request.selectedPattern);
    console.log(request);
  }
});
