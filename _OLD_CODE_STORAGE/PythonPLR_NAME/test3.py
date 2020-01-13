import plotly.plotly as py
from plotly.graph_objs import *
py.sign_in('', '')
data = Data([
    Bar(
        x=['Sivaranjani S', 'Vijayalakshmi C', 'Rajeshwari S', 'Shanthi Priscilla', 'Pandiyaraj G', 'Kamatchi S', 'MohanaPriya', 'Madhumitha G', 'Franklin Alphones Raj J', 'Akfaris Almaas', 'Biswajit Champati', 'Priya R', 'Rekha Rajasekaran', 'Sarath Kumar B', 'Jegan L', 'Karthick A', 'Mahalakshmi S', 'Ragunathan V', 'Anu S', 'Ramkumar KS', 'Uthra R'],
        y=[1640, 1394, 1390, 1313, 2166, 1521, 1078, 1543, 780, 1202, 1505, 2028, 2032, 1769, 1238, 1491, 1477, 1329, 2038, 1339, 1458],
        text=['Scuti', 'Scuti', 'Cygni', 'Scorpii', 'Scuti', 'Pollux', 'Scorpii', 'Pollux', 'Scuti', 'Pollux', 'Scorpii', 'Scorpii', 'Scuti', 'Cygni', 'Scorpii', 'Scuti', 'Scuti', 'Pollux', 'Scuti', 'Pollux', 'Pollux']
    )
])

layout = Layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)

fig = Figure(data=data, layout=layout)

plot_url = py.plot(fig, filename='transparent-background')
