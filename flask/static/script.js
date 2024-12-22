async function fetchData() {
    try {
        const response = await fetch('/api/data'); // Fetch data from the API
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        const data = await response.json(); // Parse JSON
        document.getElementById('data-display').textContent = `Count: ${data.count}`;
    } catch (error) {
        console.error('Error fetching data:', error);
        document.getElementById('data-display').textContent = 'Error loading data';
    }
}

// Fetch data every 2 seconds
setInterval(fetchData, 2000);
fetchData(); // Initial fetch
