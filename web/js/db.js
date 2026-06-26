// db.js — a tiny promise wrapper over IndexedDB for the trainer.
(function (global) {
  "use strict";

  const DB_NAME = "icv";
  const VERSION = 1;
  const STORES = ["classes", "annotations", "keypoints"];

  // Open (and, on first run or version bump, create) the database.
  function open() {
    return new Promise((resolve, reject) => {
      const req = indexedDB.open(DB_NAME, VERSION);
      req.onupgradeneeded = () => {
        const db = req.result;
        for (const name of STORES) {
          if (!db.objectStoreNames.contains(name)) {
            db.createObjectStore(name, {
              keyPath: "id",
              autoIncrement: true,
            });
          }
        }
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
  }

  function store(db, name, mode) {
    return db.transaction(name, mode).objectStore(name);
  }

  // Wrap a single IDBRequest as a Promise.
  function wrap(request) {
    return new Promise((resolve, reject) => {
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  }

  async function put(name, value) {
    const db = await open();
    return wrap(store(db, name, "readwrite").put(value));
  }

  async function getAll(name) {
    const db = await open();
    return wrap(store(db, name, "readonly").getAll());
  }

  async function remove(name, id) {
    const db = await open();
    return wrap(store(db, name, "readwrite").delete(id));
  }

  async function clear(name) {
    const db = await open();
    return wrap(store(db, name, "readwrite").clear());
  }

  global.DB = { open, put, getAll, remove, clear, STORES };

  // A generic, explicit-key store for the annotation studio: arbitrary
  // store names and (key, value) pairs, each Store its own database.
  class Store {
    constructor(dbName, stores) {
      this.dbName = dbName;
      this.stores = stores;
      this.version = 1;
    }
    _open() {
      return new Promise((resolve, reject) => {
        const req = indexedDB.open(this.dbName, this.version);
        req.onupgradeneeded = () => {
          const db = req.result;
          for (const s of this.stores) {
            if (!db.objectStoreNames.contains(s)) db.createObjectStore(s);
          }
        };
        req.onsuccess = () => resolve(req.result);
        req.onerror = () => reject(req.error);
      });
    }
    async put(storeName, key, value) {
      const db = await this._open();
      return wrap(db.transaction(storeName, "readwrite")
        .objectStore(storeName).put(value, key));
    }
    async get(storeName, key) {
      const db = await this._open();
      return wrap(db.transaction(storeName, "readonly")
        .objectStore(storeName).get(key));
    }
    async all(storeName) {
      const db = await this._open();
      return wrap(db.transaction(storeName, "readonly")
        .objectStore(storeName).getAll());
    }
    async del(storeName, key) {
      const db = await this._open();
      return wrap(db.transaction(storeName, "readwrite")
        .objectStore(storeName).delete(key));
    }
  }
  global.Store = Store;
})(window);
