setTimeout(() => {
    var demo = document.getElementById('demo-explanation');
    var button = document.getElementById('about-button')
    demo.style.display = "none";

    button.addEventListener('click', () => {
        if(demo.style.display == "none"){
            demo.style.display = "block";
        } else {
            demo.style.display = "none";
        }
    })
}, 2000) //Time definition to wait page is ready