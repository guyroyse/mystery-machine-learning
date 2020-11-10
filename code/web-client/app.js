const JINKIES_URL = 'http://localhost:3000/jinkies'

document.addEventListener('DOMContentLoaded', () => {

  /// or form elements
  let lineInput = document.querySelector('#line')
  let backendSelector = document.querySelector('#backend')
  let jinkiesButton = document.querySelector('#button')

  // get all the character images to show and hide
  // NOTE: the order here matches the order of the label encoding
  // such that the encoded result (0 for Daphne, 1 for Fred, etc.)
  // matches the index in this array
  let images = [
    document.querySelector('#daphne'),
    document.querySelector('#fred'),
    document.querySelector('#scooby'),
    document.querySelector('#shaggy'),
    document.querySelector('#velma'),
    document.querySelector('#unknown')
  ]

  // the name for the character
  let characterNameParagraph = document.querySelector('#characterName')

  // classify on click
  jinkiesButton.addEventListener('click', () => whoSaidIt())

  // allow using enter for convenience
  lineInput.addEventListener('keyup', (event) => {
    if (event.key === 'Enter') {
      whoSaidIt()
    } else {
      images.forEach(image => image.className = 'hidden')
      images[images.length  - 1].className = ''
      characterNameParagraph.textContent = "????????"
    }
  })

  async function whoSaidIt() {

    // fetch JSON via post using a JSON body with the line and the backend
    let fetchSettings = {
      method: 'POST',
      headers: { 'Content-Type': 'application/json; charset=utf-8' },
      body: JSON.stringify({
        line: lineInput.value,
        backend: backendSelector.value
      })
    }

    // get the response as JSON
    let response = await fetch(JINKIES_URL, fetchSettings)
    let result = await response.json()

    // show the image for the character
    images.forEach(image => image.className = 'hidden')
    images[result.winner.class].className = ''

    // update the text for the character name
    characterNameParagraph.textContent = result.winner.decodedClass
  }
})
