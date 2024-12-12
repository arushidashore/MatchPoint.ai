document.getElementById('inputForm').addEventListener('submit', function(event) {
  event.preventDefault(); // Prevent form submission

  const height = document.getElementById('height').value;
  const strokeType = document.getElementById('strokeType').value;

  // Simulate feedback based on user input
  let feedback = `You entered a height of ${height} cm and selected a ${strokeType} stroke.`;

  // Here you can add more logic to provide feedback based on the input
  if (height < 150) {
      feedback += " Consider improving your swing technique for better performance.";
  } else {
      feedback += " Your height is optimal for a powerful swing!";
  }

  document.getElementById('feedback').innerText = feedback;
});