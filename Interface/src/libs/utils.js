function strToJson(str) {
  let json = eval(`(${str})`);
  return json;
}

export {
  strToJson
}
