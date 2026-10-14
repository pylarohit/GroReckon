package com.rohit.georeckon.data.local

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase

@Database(entities = [TripEntity::class], version = 1, exportSchema = false)
abstract class GeoReckonDatabase : RoomDatabase() {
    abstract fun tripDao(): TripDao

    companion object {
        @Volatile
        private var INSTANCE: GeoReckonDatabase? = null

        fun getDatabase(context: Context): GeoReckonDatabase? {
            return INSTANCE ?: synchronized(this) {
                try {
                    val instance = Room.databaseBuilder(
                        context.applicationContext,
                        GeoReckonDatabase::class.java,
                        "georeckon_offline.db"
                    ).fallbackToDestructiveMigration().build()
                    INSTANCE = instance
                    instance
                } catch (e: Exception) {
                    e.printStackTrace()
                    null
                }
            }
        }
    }
}
