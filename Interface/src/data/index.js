import * as d3 from 'd3'

const cityGroup = {
    "jing-jin-ji": [],
    "chang-san-jiao": [],
    "fen-wei": [],
    "zhu-san-jiao": [],
    "cheng-yu": []
}

// const aqiColorScheme_green = d3.scaleLinear()
//     .domain([0, 50])
//     .range(['#01e400', d3.interpolateRdYlGn(0.9)])
// const aqiColorScheme_orange = d3.scaleLinear()
//     .domain([50, 150])
//     .range(['#fee493', '#ff7e00'])
// const aqiColorScheme_red = d3.scaleLinear()
//     .domain([150, 200])
//     .range(['#f34b36', '#ba1419'])
// const aqiColorScheme_purple = d3.scaleLinear()
//     .domain([200, 300])
//     .range(['#766ab0', '#4b1787'])
// const aqiColorScheme_deepRed = d3.scaleLinear()
//     .domain([300, 500])
//     .range(['#880000', '#440000'])

const aqiColorScheme_green = d3.scaleQuantize()
    .domain([0, 50])
    .range(['#66c2a4', '#41ae76', '#238b45'])

function aqiColorScheme_orange(val) {
    let aqiColorScheme_50_to_100 = d3.scaleQuantize()
        .domain([50, 100])
        .range(['#fed976', '#fec44f', '#feb24c']) // , '#fe9929'
    let aqiColorScheme_100_to_150 = d3.scaleQuantize()
        .domain([100, 150])
        // .range(['#fdae6b', '#fd8d3c', '#f16913'])
        .range(['#fe9929', '#ec7014', '#cc4c02']) // , '#993404'
    if (val >= 50 && val < 100) return aqiColorScheme_50_to_100(val)
    else if (val >= 100 && val < 150) return aqiColorScheme_100_to_150(val)
}
const aqiColorScheme_red = d3.scaleQuantize()
    .domain([150, 200])
    .range(['#fb6a4a', '#ef3b2c', '#cb181d'])
const aqiColorScheme_purple = d3.scaleQuantize()
    .domain([200, 300])
    .range(['#807dba', '#6a51a3', '#54278f', '#3f007d'])
const aqiColorScheme_deepRed = d3.scaleQuantize()
    .domain([300, 500])
    .range(['#a50f15', '#67000d'])

const aqiColorScheme = [
    aqiColorScheme_green,
    aqiColorScheme_orange,
    aqiColorScheme_red,
    aqiColorScheme_purple,
    aqiColorScheme_deepRed
]

const inputFeatureCity = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
const inputFeaturePoint = {
    air: ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3'],
    climate: ['U', 'V', 'TEMP', 'RH', 'PSFC', 'boundary_layer_h'],
    space: ['lat', 'lon', 'elevation', 'vegetation_index']
}

const outputFeatureCity = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
const outputFeaturePoint = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']

// const valColorScheme = ['#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b']
// 原来的方案：
const valColorScheme_blue = ['#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b']
const valColorScheme_red = ['#ffffff', '#a50f15']
const valColorScheme_fire = ['#fff7ec', '#fee8c8', '#fdd49e', '#fdbb84', '#fc8d59', '#ef6548', '#d7301f', '#b30000', '#7f0000', '#000000']
// const valColorScheme_fire = ['#fee8c8', '#fdd49e', '#fdbb84', '#fc8d59', '#ef6548', '#d7301f', '#b30000', '#7f0000', '#000000']  // 去掉了第一个color

// const valColorScheme_double = ['#77482e', '#ca997f', '#e5ccbd', '#f1e5de', '#ffffff', '#dfebef', '#c0d6e1', '#83aec6', '#366179']
const valColorScheme_double = ['#77482e', '#ca997f', '#e5ccbd', '#f1e5de', '#dfebef', '#c0d6e1', '#83aec6', '#366179']

const evaluate_questions = [
    
]

const compare_questions = [
  "是否能够选出一个最佳模型，哪个或哪些模型是你认为最佳的模型？比其他模型好在哪里？",
  "所关注的模型，When and where 会发生何种误差",
  "模型在什么情况下会发生failure，failure(对实际应用的)影响有多大？",
  "所关注的模型，是否能够预测出一些演化重要的演化模式？"

]

export {
    cityGroup,
    aqiColorScheme,
    valColorScheme_blue,
    valColorScheme_red,
    valColorScheme_fire,
    valColorScheme_double,
    inputFeatureCity,
    outputFeatureCity,
    inputFeaturePoint,
    outputFeaturePoint
}