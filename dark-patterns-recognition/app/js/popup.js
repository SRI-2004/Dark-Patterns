window.onload = function () {
  chrome.tabs.query({ currentWindow: true, active: true }, function (tabs) {
    chrome.tabs.sendMessage(tabs[0].id, { message: "popup_open" });
  });

  document.getElementsByClassName("analyze-button")[0].onclick = function () {
    console.log("analyze")
    chrome.tabs.query({ currentWindow: true, active: true }, function (tabs) {
      chrome.tabs.sendMessage(tabs[0].id, { message: "analyze_site" });
    });
  };

  document.getElementsByClassName("link")[0].onclick = function (event) {
    event.preventDefault(); // Prevent the default behavior of opening a new tab
    chrome.tabs.create({
      url: document.getElementsByClassName("link")[0].getAttribute("href"),
    });
  };

  populateDropdown("patternDropdown");

  function toggleFeedbackForm() {
    const feedbackForm = document.getElementById("feedbackForm");
    feedbackForm.style.display = feedbackForm.style.display === 'none' ? 'block' : 'none';
    
    const feedbackBtn = document.getElementById("feedbackBtn");
    feedbackBtn.innerText = feedbackBtn.innerText === 'Feedback' ? 'Close Feedback' : 'Feedback';
}

  document.getElementById("feedbackBtn").onclick = function (event) {
    event.preventDefault(); // Prevent the default behavior of opening a new tab
    toggleFeedbackForm();
  };  

document.getElementsByClassName("feedback")[0].onclick = function () {
    console.log("feedback click");
    const feedbackValue = document.getElementById("feedbackInput").value;
    const selectedPattern = document.getElementById("patternDropdown").value;
    const feedbackTypeButtons = document.querySelectorAll('[name="feedbackType"]');
    let feedbackType;

    // Loop through the buttons to find the one that is clicked
    feedbackTypeButtons.forEach(button => {
        if (button.classList.contains('selected')) {
            feedbackType = button.value;
        }
    });

    if (feedbackValue && selectedPattern && feedbackType) {
        chrome.tabs.query({ currentWindow: true, active: true }, function (tabs) {
            chrome.tabs.sendMessage(tabs[0].id, {
                message: "feedback",
                feedbackValue: feedbackValue,
                selectedPattern: selectedPattern,
                feedbackType: feedbackType
            });
        });

        // Clear input fields
        document.getElementById("feedbackInput").value = "";
        document.getElementById("patternDropdown").value = "";

        // Remove selection from buttons
        feedbackTypeButtons.forEach(button => {
            button.classList.remove('selected');
        });

        // Alert to the user
        toggleFeedbackForm()
        alert("Feedback received successfully!");
    } else {
        // Alert if feedback, pattern, or feedback type is not provided
        alert("Please enter feedback, select a pattern, and choose feedback type.");
    }
};




// Add event listeners to the buttons to toggle the selection
document.querySelectorAll('[name="feedbackType"]').forEach(button => {
    button.addEventListener('click', function() {
        // Remove blue background color from all buttons
        document.querySelectorAll('[name="feedbackType"]').forEach(btn => {
            btn.style.backgroundColor = ''; // Reset background color
            btn.classList.remove('selected');
        });
        // Set blue background color to the clicked button
        this.style.backgroundColor = 'grey';
        this.classList.add('selected');
    });
});




};

chrome.runtime.onMessage.addListener(function (request, sender, sendResponse) {
  if (request.message === "update_dark_pattern_counts") {
    updateCounts(request.counts);
  }
});

function updateCounts(counts) {
  console.log("Updating counts:", counts);

  updateDarkPatternCount("Sneaking", counts.Sneaking);
  updateDarkPatternCount("Urgency", counts.Urgency);
  updateDarkPatternCount("Misdirection", counts.Misdirection);
  updateDarkPatternCount("Social Proof", counts["Social Proof"]);
  updateDarkPatternCount("Scarcity", counts.Scarcity);
  updateDarkPatternCount("Obstruction", counts.Obstruction);
  updateDarkPatternCount("Forced Action", counts["Forced Action"]);
}

function updateDarkPatternCount(pattern, count) {
  let element = document.querySelector(`.${pattern}-number`);
  if (element) {
    console.log(`Updating count for ${pattern}: ${count}`);
    if (count > 0)
    {
        element.textContent = `${pattern} pattern: ${count} occurrences`;
    }
  }
}

// Function to populate dropdown options
function populateDropdown(dropdownId) {
    const dropdown = document.getElementById(dropdownId);
    
    // List of pattern descriptions
    const descriptions = [
        "Sneaking",
        "Urgency",
        "Misdirection",
        "Social Proof",
        "Scarcity",
        "Obstruction",
        "Forced Action",
    ];

    descriptions.forEach(option => {
        const optionElement = document.createElement("option");
        optionElement.value = option;
        optionElement.text = option;
        dropdown.add(optionElement);
    });
}


document.getElementById("analyze-button").onclick = function (event) {
    event.preventDefault(); // Prevent the default behavior of opening a new tab
    const darkpatterncontainer = document.getElementById("container");
    darkpatterncontainer.style.display = darkpatterncontainer.style.display === 'none' ? 'block' : 'none';
    
  }; 