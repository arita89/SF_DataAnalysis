from modules import *

## Tokens ##
# rapidApi keys
RAPIDAPI_KEY= os.environ.get("RAPIDAPI_KEY")
RAPIDAPI_HOST= os.environ.get("RAPIDAPI_HOST")
headers = {
      'x-rapidapi-key': RAPIDAPI_KEY,
      'x-rapidapi-host': RAPIDAPI_HOST,
      }

# Socrata keys 
myemail= os.environ.get("myemail")
mypsw= os.environ.get("mypsw")
SocrataToken = os.environ.get("SocrataToken")


## Default Var ##
# colors 
colors = list(dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS).values())
# Use calendar library for abbreviations and order
dd=dict((enumerate(calendar.month_abbr)))
hh = {k: f"{k}-{k+1}" for k in range(0,24)}
hhsimple = {k: f"{k}" for k in range(0,24)}

daytoInt = {"Monday": 0,
            "Tuesday": 1,
            "Wednesday":2,
            'Thursday':3, 
            'Friday':4, 
            'Saturday':5, 
            'Sunday':6
            }

from branca.element import Template, MacroElement

## courtesy of stackoverflow https://stackoverflow.com/questions/64382103/how-to-add-legend-on-html-with-folium


template = """
{% macro html(this, kwargs) %}

<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>jQuery UI Draggable - Default functionality</title>
  <link rel="stylesheet" href="//code.jquery.com/ui/1.12.1/themes/base/jquery-ui.css">

  <script src="https://code.jquery.com/jquery-1.12.4.js"></script>
  <script src="https://code.jquery.com/ui/1.12.1/jquery-ui.js"></script>
  
  <script>
  $( function() {
    $( "#maplegend" ).draggable({
                    start: function (event, ui) {
                        $(this).css({
                            right: "auto",
                            top: "auto",
                            bottom: "auto"
                        });
                    }
                });
});

  </script>
</head>
<body>

 
<div id='maplegend' class='maplegend' 
    style='position: absolute; z-index:9999; border:2px solid grey; background-color:rgba(255, 255, 255, 0.8);
     border-radius:6px; padding: 10px; font-size:14px; right: 20px; bottom: 20px;'>
     
<div class='legend-title'>Legend</div>
<div class='legend-scale'>
  <ul class='legend-labels'>
    <li><span style='background:blue;opacity:0.7;'></span>PROSTITUTION</li>
    <li><span style='background:green;opacity:0.7;'></span>DRUG/NARCOTIC</li>
    <li><span style='background:red;opacity:0.7;'></span>DRIVING UNDER THE INFLUENCE</li>

  </ul>
</div>
</div>
 
</body>
</html>

<style type='text/css'>
  .maplegend .legend-title {
    text-align: left;
    margin-bottom: 5px;
    font-weight: bold;
    font-size: 90%;
    }
  .maplegend .legend-scale ul {
    margin: 0;
    margin-bottom: 5px;
    padding: 0;
    float: left;
    list-style: none;
    }
  .maplegend .legend-scale ul li {
    font-size: 80%;
    list-style: none;
    margin-left: 0;
    line-height: 18px;
    margin-bottom: 2px;
    }
  .maplegend ul.legend-labels li span {
    display: block;
    float: left;
    height: 16px;
    width: 30px;
    margin-right: 5px;
    margin-left: 0;
    border: 1px solid #999;
    }
  .maplegend .legend-source {
    font-size: 80%;
    color: #777;
    clear: both;
    }
  .maplegend a {
    color: #777;
    }
</style>
{% endmacro %}"""

print ("Tokens and default var imported")