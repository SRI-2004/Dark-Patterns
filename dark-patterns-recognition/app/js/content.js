const endpoint = "http:/127.0.0.1:5000/";
const descriptions = {
  "Sneaking": "Coerces users to act in ways that they would not normally act by obscuring information.",
  "Urgency": "Places deadlines on things to make them appear more desirable",
  "Misdirection": "Aims to deceptively incline a user towards one choice over the other.",
  "Social Proof": "Gives the perception that a given action or product has been approved by other people.",
  "Scarcity": "Tries to increase the value of something by making it appear to be limited in availability.",
  "Obstruction": "Tries to make an action more difficult so that a user is less likely to do that action.",
  "Forced Action": "Forces a user to complete extra, unrelated tasks to do something that should be simple.",
};

function scrape() {
  if (document.getElementById("insite_count")) {
    return;
  }

  // aggregate all DOM elements on the page
  let elements = segments(document.body);
  let filtered_elements = [];
  let patternCounts = {};
  let siteUrl = window.location.href;
  let text;

  for (let i = 0; i < elements.length; i++) {
    if (elements[i].innerText) {
      console.log(elements[i]);
      text = elements[i].innerText.trim().replace(/\t/g, " ");
    } else {
      text = "";
    }
    if (text.length == 0) {
      continue;
    }
    filtered_elements.push(text);
  }

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

      let dp_counts = {}; // To store counts for each pattern
      let element_index = 0;
      let text;
      for (let i = 0; i < elements.length; i++) {
        if (elements[i].innerText) {
          console.log(elements[i]);
          text = elements[i].innerText.trim().replace(/\t/g, " ");
        } else {
          text = "";
        }

        if (text.length == 0) {
          continue;
        }

        if (json.result[i] !== "Not Dark") {
          dp_counts[json.result[i]] = (dp_counts[json.result[i]] || 0) + 1;
          highlight(elements[element_index], json.result[i]);
        }
        element_index++;
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
}
function highlight(element, type) {
  element.classList.add("insite-highlight");

  if (type && type.trim() !== "") {
    // Remove spaces from the type variable
    const tempType = type.replace(/\s+/g, "");
    // Add the modified type to the class list of the element
    element.classList.add(tempType);
  }

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
  element.addEventListener("mouseover", function () {
    showPopup(type);
  });

  element.addEventListener("mouseout", function () {
    hidePopup();
  });
}

function showPopup(type) {
  const popup = document.getElementById("dark-pattern-popup");
  const description = descriptions[type];

  popup.innerHTML = `
      <div class="modal-header">
        <h1>${type} Pattern</h1>
      </div>
      <div class="modal-content">
        ${description}
      </div>
    `;

  popup.style.display = "block";
}

function hidePopup() {
  const popup = document.getElementById("dark-pattern-popup");
  popup.style.display = "none";
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
