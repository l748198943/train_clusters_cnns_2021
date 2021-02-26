<script>
  import { onMount } from "svelte";

  let tableData = [];
  let textData = [];
  let tableHeader = ['Prediction / Label','Text', 'Dimentions'];
  let keyWords = [];
  let text = [];
  let contributions = 0.0;
  let antonyms = [];
  let bt = "<button>Download</button>"
  let currentWord = 0

   function addOnClick() {
	antonyms = textData[keyWords[2]].Antonyms;
   }

  onMount(async () => {
    const res2 = await fetch(`http://localhost:8002/gradxinput-203.json`); 
    textData = await res2.json();
    text = textData["Text"]
    keyWords = Object.keys(textData).slice(3);
    antonyms = textData[keyWords[0]].Antonyms;
    contributions = textData[keyWords[0]].Contribution;
    for (var i=0;i<keyWords.length;i++)
	{ 
            var pos = Object.values(textData[keyWords[i]]["Position"]);
	    var sent_pos = pos[0];
            var word_pos = pos[1];
	    console.log(pos);
            var sent_splitted = text[sent_pos].split(" ");
            sent_splitted[word_pos] = "<button\tstyle='border-radius:15px';\tid=button_"+i.toString()+">" + sent_splitted[word_pos] +"</button>";
            text[sent_pos] = sent_splitted.join(" ");
	}
    
  });

    function initButtons() {
        console.log('');
        for(var i=0;i<keyWords.length;i++){
                console.log(document.getElementById("button_"+i.toString()));
	  	var el = document.getElementById("button_"+i.toString());
                if(el){
                  el.addEventListener("click", function() {
                     var buttonId = parseInt(event.target.id.split("_")[1]);
                     event.target.style.color = 'red'; 
                     for(var j=0;j<keyWords.length;j++){
                         var el2 = document.getElementById("button_"+j.toString());
                         if(el2 && el2.id!=event.target.id){
                            el2.style.color = 'black';
                         }
                     }
                     currentWord = buttonId;
                     antonyms = textData[keyWords[buttonId]].Antonyms;
                     contributions = textData[keyWords[buttonId]].Contribution;
		     }, false);
                  }
       }

    }

</script>

<style>
  .container {
    max-width: 1140px;
    margin: auto;
  }
  .header {
    display: flex;
    justify-content: space-between;
    display: flex;
    justify-content: space-between;
    background: orange;
    padding: 10px;
  }
  #rcorners2 {
    border-radius: 25px;
    border: 2px solid #8AC007;
    padding: 8px;   
  }
  .progress-bar{
    width:80%;
    height: 5px;
    background-color: #f9e1e3;
    border-radius: 3px;
  }
  .progress{
    width: 60%;
    height: 100%;
    background-color: #e46a70;
    border-radius: 3px;
  }
  table {
    font-family: arial, sans-serif;
    border-collapse: collapse;
    width: 100%;
  }

  td,
  th {
    border: 1px solid #dddddd;
    text-align: left;
    padding: 18px;
  }

  tr:nth-child(even) {
    background-color: #dddddd;
  }
  button {
    border: none; /* Remove borders */
    color: white; /* Add a text color */
    padding: 14px 28px; /* Add some padding */
    cursor: pointer; /* Add a pointer cursor on mouse-over */
    background-color: #4caf50;
    height: fit-content;
  }
  h1 {
    margin: 0px;
  }
</style>

<div class="container">
  <div class="header">
    <h1>Evaluate gradient values</h1>
    <button on:click={initButtons}>Initialize</button>
    <input type="range" id="range1" min="0" max="100" step="5" onchange="b.value=this.value"/>
    <output id="b" for="range1" ></output>

  </div>

  <div class="main">
    <table>
      <thead>
        <tr>
          {#each tableHeader as header}
            <th>{header}</th>
          {/each}
        </tr>
      </thead>
      <tbody>
          <tr>
            <td style="width: 80px">{textData.Prediction+' / '+ textData.Label}</td>
            <td style="width: 300px">
              {#each text.slice(0,10) as sent}
		<p id="rcorners2">{@html sent}</p> 
              {/each}
	    </td>
            <td>
              <h2>{"Sum of contributions of the word: "+ parseFloat(contributions).toFixed(3)}</h2>
              <h3>Normalized contribution values:</h3>
              {#each Object.keys(antonyms).slice(0,10) as dimName}
     		<h3>{dimName+": "+parseFloat(antonyms[dimName]).toFixed(2)}</h3>
     		<div class="progress-bar" style="width: 80%">
        	  <div class="progress" style={"width:" + parseFloat((antonyms[dimName])/36*100).toString()+"%"}>
        	  </div>
    		</div>
	      {/each}
            </td>
          </tr>
      </tbody>
    </table>

  </div>
</div>
