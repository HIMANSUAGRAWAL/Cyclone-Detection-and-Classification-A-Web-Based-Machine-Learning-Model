document.getElementById("cycloneForm").addEventListener("submit", function(event) {
    event.preventDefault();

    // Gather form data
    const formData = new FormData(this);

    // Make a POST request to /classify endpoint
    fetch("/classify", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        // Display the results
        document.getElementById("cyclone_type").textContent = `Cyclone Type: ${data.cyclone_type}`;
        document.getElementById("description").textContent = `Description: ${data.description}`;
    })
    .catch(error => console.error("Error:", error));
});
