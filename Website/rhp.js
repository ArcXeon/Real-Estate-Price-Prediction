function onClickedEstimatePrice() {
    console.log("Estimate price button clicked");
    var area = document.getElementById("uiArea");
    var bhk = document.getElementById("uibhk");
    var bathrooms = document.getElementById("uiBathrooms");
    var location = document.getElementById("uiLocations");
    var estPrice = document.getElementById("uiEstimatedPrice");

    var url = "http://127.0.0.1:5000/predict_home_price";


    $.post(url, {
        total_sqft: parseFloat(area.value),
        bhk: parseInt(bhk.value),
        bath: parseInt(bathrooms.value),
        location: location.value
    },function(data, status) {
        console.log(data.estimated_price);
        estPrice.innerHTML = "<h2>" + data.estimated_price.toString() + " Lakh</h2>";
        console.log(status);
    });
    }    

    function onPageLoad() {
        console.log( "document loaded" );
        var url = "http://127.0.0.1:5000/get_location_names"; 
        $.get(url,function(data, status) {
            console.log("got response for get_location_names request");
            if(data) {
                var locations = data.locations;
                var uiLocations = document.getElementById("uiLocations");
                $('#uiLocations').empty();
                for(var i in locations) {
                    var opt = new Option(locations[i]);
                    $('#uiLocations').append(opt);
                }
            }
        });
      }
      
      window.onload = onPageLoad;