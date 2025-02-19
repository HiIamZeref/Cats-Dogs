import { api } from "./Api.ts";

export const makePrediction = async function (imageForm: FormData) {
  return await api.post("/make_predictions", imageForm, {
    headers: {
      "Content-Type": "multipart/form-data",
    },
  });
};
