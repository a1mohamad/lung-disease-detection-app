(function () {
  var path = window.location.pathname;
  var githubPagesBase = "/apps/lung-disease-detection";
  var fastApiBase = "/ui";
  var basePath = "";

  if (path === githubPagesBase || path.indexOf(githubPagesBase + "/") === 0) {
    basePath = githubPagesBase;
  } else if (path === fastApiBase || path.indexOf(fastApiBase + "/") === 0) {
    basePath = fastApiBase;
  }

  window.APP_CONFIG = {
    API_BASE_URL: "https://a1mohamadd-lung-disease-detection-api.hf.space",
    BASE_PATH: basePath
  };
})();
