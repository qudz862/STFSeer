import { defineStore } from 'pinia'

export const useTestStore = defineStore('test', {
  state: () => ({
    age: 30
  }),
  getters: {
    realAge: (state) => state.age + 3
  },
  actions: {
    addAge() {
      this.age++
    }
  }
})